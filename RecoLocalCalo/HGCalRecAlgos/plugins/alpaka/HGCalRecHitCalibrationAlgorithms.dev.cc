#include <cstddef>

// CMSSW imports
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Alpaka imports
//#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

// HGCal imports
#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalRecHitCalibrationAlgorithms.h"
#include "DataFormats/HGCalDigi/interface/HGCalRawDataDefinitions.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  // @short set RecHit flags based on DIGI flags (data corruption, zero-suppression)
  struct HGCalRecHitCalibrationKernel_flagRecHits {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalSoARecHitsDeviceCollection::View recHits,
                                  HGCalDigiDevice::ConstView digis,
                                  HGCalCalibParamDevice::ConstView calibs) const {
      for (auto idx : uniform_elements(acc, digis.metadata().size())) {
        bool calibvalid = calibs[idx].valid();
        auto digiflags = digis[idx].flags();
        bool isAvailable((digiflags != ::hgcal::DIGI_FLAG::Invalid) &&
                         (digiflags != ::hgcal::DIGI_FLAG::NotAvailable) && calibvalid);
        bool isToAavailable((digiflags != ::hgcal::DIGI_FLAG::ZS_ToA) &&
                            (digiflags != ::hgcal::DIGI_FLAG::ZS_ToA_ADCm1));
        recHits[idx].flags() =
            (!isAvailable) * HGCalRecHitFlags::kEnergyInvalid + (!isToAavailable) * HGCalRecHitFlags::kTimeInvalid;
      }
    }
  };

  // @short subtract pedestals, linearize ADC & TOT to charge (fC), and convert to energy (GeV)
  // energy: linear CM correction; energy_variations(CM_ALT): DNN CM correction
  struct HGCalRecHitCalibrationKernel_adcToEnergy {
    bool cmAltAsNominal_;  // swap: energy uses the DNN, energy_variations(CM_ALT) the linear CM
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalSoARecHitsDeviceCollection::View recHits,
                                  HGCalDigiDevice::ConstView digis,
                                  HGCalCalibParamDevice::ConstView calibs,
                                  HGCalCMCorrectionDeviceCollection::ConstView corrections) const {
      auto ped_sub = [&](uint32_t adc, float adc_ped) { return adc - adc_ped; };
      auto cm_linear = [&](uint32_t cm, float cm_slope, float cm_ped) {
        return cm_slope * (0.5f * float(cm) - cm_ped);
      };
      // before there was just one python lambda like function adc_denoise but now there are two denoise functions.
      auto denoise_linear = [&](float pedsub, float pedsub_m1, float cmf, float bxm1_slope) {
        return (pedsub - cmf - bxm1_slope * (pedsub_m1 - cmf));
      };
      auto denoise_nn = [&](float pedsub, float nn) { return pedsub - nn; };  // no BX-1 term

      auto tot_linearization =
          [&](uint32_t tot, float tot_lin, float tot2adc, float tot_ped, float tot_p0, float tot_p1, float tot_p2) {
            bool isLin(tot > tot_lin);
            bool isNotLin(!isLin);
            return isLin * (tot2adc * (tot - tot_ped)) + isNotLin * (tot_p0 + tot_p1 * tot + tot_p2 * tot * tot);
          };

      for (auto idx : uniform_elements(acc, digis.metadata().size())) {
        auto calib = calibs[idx];
        bool calibvalid = calib.valid();
        auto digi = digis[idx];
        auto digiflags = digi.flags();
        bool isAvailable((digiflags != ::hgcal::DIGI_FLAG::Invalid) &&
                         (digiflags != ::hgcal::DIGI_FLAG::NotAvailable) && calibvalid);
        bool useTOT((digi.tctp() == 3) && isAvailable);
        bool useADC(!useTOT && isAvailable);

        float pedsub = ped_sub(digi.adc(), calib.ADC_ped());
        float pedsub_m1 = ped_sub(digi.adcm1(), calib.ADC_ped());
        float adc_linear = denoise_linear(
            pedsub, pedsub_m1, cm_linear(digi.cm(), calib.CM_slope(), calib.CM_ped()), calib.BXm1_slope());
        bool useNN(corrections[idx].valid());
        float adc_cm_alt = useNN ? denoise_nn(pedsub, corrections[idx].correction()) : adc_linear;

        float tot_counts = tot_linearization(digi.tot(),
                                             calib.TOT_lin(),
                                             calib.TOTtoADC(),
                                             calib.TOT_ped(),
                                             calib.TOT_P0(),
                                             calib.TOT_P1(),
                                             calib.TOT_P2());
        // Before you will see lin_counts was set to using the linear method, if cmAltAsNominal_ is passed in as an arg
        // we use this as the default and then the linear as alternative (cm_alt).
        auto lin_counts = useADC * (cmAltAsNominal_ ? adc_cm_alt : adc_linear) + useTOT * tot_counts;
        auto lin_counts_cm_alt = useADC * (cmAltAsNominal_ ? adc_linear : adc_cm_alt) + useTOT * tot_counts;

        // after denoising/linearization apply the MIP & EM scale to convert to energy (GeV)
        // noise is also saved in units of GeV
        recHits[idx].mipEnergy() = lin_counts * calib.MIPS_scale();
        recHits[idx].energy() = calib.EM_scale() * calib.MIPS_scale() * lin_counts;
        // Energy variation should be easy to comprehend.
        recHits[idx].energy_variations()(hgcalreco::Variation::CM_ALT) =
            calib.EM_scale() * calib.MIPS_scale() * lin_counts_cm_alt;
        recHits[idx].energy_variations()(hgcalreco::Variation::SCALE_ALT) = recHits[idx].energy();
        recHits[idx].sigmaNoise() = calib.EM_scale() * calib.MIPS_scale() * calib.Noise();
      }
    }
  };

  // @short apply INL and timewalk corrections, and convert TOA to time (ns), shifted by 12.5 ns when there is a timewalk
  // calibration. A rechit has a time if it is available, its TOA was read out and fired (toa > 0) and it has a noise;
  // the others get time -99 and timeError -1. The time error follows from the S/N of the rechit.
  struct HGCalRecHitCalibrationKernel_toaToTime {
    HGCalRecHitTimeResolution timeResolution;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalSoARecHitsDeviceCollection::View recHits,
                                  HGCalDigiDevice::ConstView digis,
                                  HGCalCalibParamDevice::ConstView calibs) const {
      auto toa_inl_corr = [&](uint32_t toa, hgcalrechit::Vector32f ctdc_p, hgcalrechit::Vector8f ftdc_p) {
        auto gray = toa / 256;
        auto ptdc = toa % 256;
        auto ctdc = ptdc / 8;
        auto ftdc = ptdc % 8;
        auto ctdc_corr = uint(ctdc - ctdc_p[ctdc]) % 32;
        auto ftdc_corr = uint(ftdc - ftdc_p[ftdc]) % 8;
        return (ftdc_corr + 8 * ctdc_corr + 256 * gray) % 1024;
      };

      // Time walk: TOA_TW = (c, m, p2) of the fit toa = c + m / (ADC - p2) in HGCalTWAnalysis.py (hgcal-comm), with ADC
      // the linearised ADC counts of the rechit. In TDC counts, kept as a float: the correction is not a whole number of
      // counts and can be larger than the TOA
      auto toa_tw_corr = [&](float toa, float adc, hgcalrechit::Vector3f p) {
        return toa - ((adc > p[2]) ? (p[0] + p[1] / (adc - p[2])) : 0.f);
      };

      // TOAtops is in ps per TDC count; the rechit time is in ns, like the time resolution and the layer clusters
      constexpr float kTOAtons = hgcalrechit::TOAtops * 1e-3f;
      // the time walk subtracts c, the TOA at infinite ADC, so the times sit around 0; shift them by half a bunch
      // crossing so that they lie within the 25 ns TOA window again. Not applied without a timewalk calibration (c = 0)
      constexpr float kTimeOffset = 12.5f;  // ns

      for (auto idx : uniform_elements(acc, digis.metadata().size())) {
        auto calib = calibs[idx];
        bool calibvalid = calib.valid();
        auto digi = digis[idx];
        auto digiflags = digi.flags();
        bool isAvailable((digiflags != ::hgcal::DIGI_FLAG::Invalid) &&
                         (digiflags != ::hgcal::DIGI_FLAG::NotAvailable) && calibvalid);
        bool isToAavailable((digiflags != ::hgcal::DIGI_FLAG::ZS_ToA) &&
                            (digiflags != ::hgcal::DIGI_FLAG::ZS_ToA_ADCm1));
        auto recHit = recHits[idx];
        // with full readout the TOA is always sent, and is 0 if it did not fire
        if (!(isAvailable && isToAavailable && digi.toa() > 0 && recHit.sigmaNoise() > 0.f)) {
          recHit.time() = HGCalRecHitTimeResolution::kNoTime;
          recHit.timeError() = HGCalRecHitTimeResolution::kNoTimeError;
          for (uint8_t k = 0; k < hgcalreco::Variation::Size; ++k) {
            recHit.time_variations()(k) = HGCalRecHitTimeResolution::kNoTime;
            recHit.timeError_variations()(k) = HGCalRecHitTimeResolution::kNoTimeError;
          }
          continue;
        }
        // INL correction
        float const toa = toa_inl_corr(digi.toa(), calib.TOA_CTDC(), calib.TOA_FTDC());
        // ADC counts of the nominal energy and of each variation, for the timewalk correction
        float const gevPerADC = calib.EM_scale() * calib.MIPS_scale();
        float const adc = calib.MIPS_scale() > 0.f ? recHit.mipEnergy() / calib.MIPS_scale() : 0.f;
        float const offset = calib.TOA_TW()[0] != 0.f ? kTimeOffset : 0.f;
        // timewalk correction, TOA to ns, offset; the variations redo it with their ADC counts
        recHit.time() = toa_tw_corr(toa, adc, calib.TOA_TW()) * kTOAtons + offset;
        recHit.timeError() = timeResolution(recHit.energy(), recHit.sigmaNoise());
        for (uint8_t k = 0; k < hgcalreco::Variation::Size; ++k) {
          float const energy = recHit.energy_variations()(k);
          float const adcK = gevPerADC > 0.f ? energy / gevPerADC : 0.f;
          recHit.time_variations()(k) = toa_tw_corr(toa, adcK, calib.TOA_TW()) * kTOAtons + offset;
          recHit.timeError_variations()(k) = timeResolution(energy, recHit.sigmaNoise());
        }
      }
    }
  };

  // @short sum the energy of the RecHits in the calibration and surrounding cells
  // see https://github.com/cms-sw/cmssw/pull/47829
  // The time is the average of the times of the two cells, weighted by their energies before the merge (a cell without
  // a time or with a negative energy has weight 0); without weight there is no time. The time error follows from the
  // merged energy and noise.
  struct HGCalRecHitCalibrationKernel_handleCalibCell {
    HGCalRecHitTimeResolution timeResolution;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalSoARecHitsDeviceCollection::View recHits,
                                  HGCalDigiDevice::ConstView digis,
                                  HGCalCalibParamDevice::ConstView calibs,
                                  HGCalMappingCellParamDevice::ConstView cell_maps,
                                  HGCalDenseIndexInfoDevice::ConstView index) const {
      auto time_weight = [](float timeError, float energy) {
        return (timeError >= 0.f && energy > 0.f) ? energy : 0.f;
      };

      for (auto idx : uniform_elements(acc, digis.metadata().size())) {
        auto cellIndex = index[idx].cellInfoIdx();

        //only applies to Si
        bool isSiPM(cell_maps[cellIndex].isSiPM());
        if (isSiPM)
          continue;

        auto calib = calibs[idx];
        bool calibvalid = calib.valid();
        auto digi = digis[idx];
        auto digiflags = digi.flags();
        bool isAvailable((digiflags != ::hgcal::DIGI_FLAG::Invalid) &&
                         (digiflags != ::hgcal::DIGI_FLAG::NotAvailable) && calibvalid);

        bool isCalibCell(cell_maps[cellIndex].iscalib());
        int offset = cell_maps[cellIndex].caliboffset();  // calibration-to-surrounding cell offset
        bool is_surr_cell((offset != 0) && isAvailable && isCalibCell);

        // effectively operate only on the cell that surrounds the calibration cells
        if (!is_surr_cell) {
          continue;
        }

        recHits[idx + offset].flags() = HGCalRecHitFlags::kNormal;

        // time weights of the surrounding (idx + offset) and calibration (idx) cells, from the energies before the merge
        float const w_surr = time_weight(recHits[idx + offset].timeError(), recHits[idx + offset].energy());
        float const w_calib = time_weight(recHits[idx].timeError(), recHits[idx].energy());
        bool const hasTime = (w_surr + w_calib) > 0.f;
        float const time =
            hasTime ? (w_surr * recHits[idx + offset].time() + w_calib * recHits[idx].time()) / (w_surr + w_calib)
                    : HGCalRecHitTimeResolution::kNoTime;
        // the same with the energies of each variation; a variation has a time if the nominal one has, and takes the
        // nominal time if its energies give no weight
        for (uint8_t k = 0; k < hgcalreco::Variation::Size; ++k) {
          float const wk_surr = time_weight(recHits[idx + offset].timeError_variations()(k),
                                            recHits[idx + offset].energy_variations()(k));
          float const wk_calib =
              time_weight(recHits[idx].timeError_variations()(k), recHits[idx].energy_variations()(k));
          recHits[idx + offset].time_variations()(k) = (hasTime && (wk_surr + wk_calib) > 0.f)
                                                           ? (wk_surr * recHits[idx + offset].time_variations()(k) +
                                                              wk_calib * recHits[idx].time_variations()(k)) /
                                                                 (wk_surr + wk_calib)
                                                           : time;
        }

        bool is_negative_surr_energy(recHits[idx + offset].energy() < 0);
        auto negative_energy_correction = (-1.0 * recHits[idx + offset].energy()) * is_negative_surr_energy;
        auto negative_mipEnergy_correction = (-1.0 * recHits[idx + offset].mipEnergy()) * is_negative_surr_energy;

        recHits[idx + offset].energy() += (negative_energy_correction + recHits[idx].energy());
        recHits[idx + offset].mipEnergy() += (negative_mipEnergy_correction + recHits[idx].mipEnergy());
        for (uint8_t k = 0; k < hgcalreco::Variation::Size; ++k) {
          float surr = recHits[idx + offset].energy_variations()(k);
          bool is_negative_surr(surr < 0);
          recHits[idx + offset].energy_variations()(k) =
              surr + (-1.0f * surr) * is_negative_surr + recHits[idx].energy_variations()(k);
        }
        float quadratic_noise_sum = sqrt(recHits[idx].sigmaNoise() * recHits[idx].sigmaNoise() +
                                         recHits[idx + offset].sigmaNoise() * recHits[idx + offset].sigmaNoise());
        recHits[idx + offset].sigmaNoise() = quadratic_noise_sum;

        recHits[idx + offset].time() = time;
        recHits[idx + offset].timeError() = hasTime
                                                ? timeResolution(recHits[idx + offset].energy(), quadratic_noise_sum)
                                                : HGCalRecHitTimeResolution::kNoTimeError;
        for (uint8_t k = 0; k < hgcalreco::Variation::Size; ++k) {
          recHits[idx + offset].timeError_variations()(k) =
              hasTime ? timeResolution(recHits[idx + offset].energy_variations()(k), quadratic_noise_sum)
                      : HGCalRecHitTimeResolution::kNoTimeError;
        }
      }
    }
  };

  // add metadata to rechits
  struct HGCalRecHitCalibrationKernel_metaData {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalSoARecHitsDeviceCollection::View recHits,
                                  HGCalMappingModuleParamDevice::ConstView mapmod,
                                  HGCalDenseIndexInfoDevice::ConstView index) const {
      for (auto idx : uniform_elements(acc, recHits.metadata().size())) {
        auto denseIdx = index[idx];
        auto recHit = recHits[idx];
        if (mapmod[denseIdx.modInfoIdx()].isSiPM()) {
          recHit.dim1() = denseIdx.eta();
          recHit.dim2() = denseIdx.phi();
        }  // else, isSilicon == true and eta phi values will not be used
        else {
          recHit.dim1() = denseIdx.x();
          recHit.dim2() = denseIdx.y();
        }
        recHit.dim3() = denseIdx.z();
        recHit.recHitIndex() = idx;
        recHit.detid() = denseIdx.detid();
        recHit.layer() = denseIdx.layer();
      }
    }
  };

  // @short print RecHits for debugging
  struct HGCalRecHitCalibrationKernel_printRecHits {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, HGCalSoARecHitsDeviceCollection::ConstView view, int size) const {
      for (int i = 0; i < size; ++i) {
        auto const& recHit = view[i];
        printf(
            "i=%d\t dim1=%f\t dim2=%f\t dim3=%f\t sigmaNoise=%f\t layer=%d\t detid=%u\t mipEnergy=%f\t "
            "energy=%f\t time=%f\t timeError=%f\t flags=%d\n",
            i,
            recHit.dim1(),
            recHit.dim2(),
            recHit.dim3(),
            recHit.sigmaNoise(),
            recHit.layer(),
            recHit.detid(),
            recHit.mipEnergy(),
            recHit.energy(),
            recHit.time(),
            recHit.timeError(),
            recHit.flags());
      }
    }
  };

  // Step 1 of the grouping by layer (see hgcalrechitlayers in the header).
  // Selects the rechits that are kept: sidx[0..nsel-1] gets their indices, in no particular order, and
  // layerCounts[slot] the number kept in each layer slot.
  struct HGCalRecHitCalibrationKernel_selectRecHits {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  int32_t* __restrict__ nsel,
                                  int32_t* __restrict__ sidx,
                                  uint32_t* __restrict__ layerCounts,
                                  HGCalSoARecHitsDeviceCollection::ConstView recHits,
                                  HGCalMappingCellParamDevice::ConstView cell_maps,
                                  HGCalDenseIndexInfoDevice::ConstView index,
                                  double k_noise = 0.) const {
      for (auto idx : uniform_elements(acc, recHits.metadata().size())) {
        auto const& recHit = recHits[idx];
        bool const goodQuality = !recHit.flags();
        bool const aboveNoise = recHit.energy() > k_noise * recHit.sigmaNoise();  // zero suppression
        bool const connectedCell = cell_maps[index[idx].cellInfoIdx()].t() > 0;   // not a calibration cell
        if (!(goodQuality && aboveNoise && connectedCell))
          continue;

        // atomicAdd returns the value before the increment, so every kept rechit gets its own place in sidx
        int32_t const place = alpaka::atomicAdd(acc, nsel, 1);
        sidx[place] = idx;
        uint32_t const slot = hgcalrechitlayers::slot(recHit.layer(), recHit.dim3());
        alpaka::atomicAdd(acc, &layerCounts[slot], 1u);
      }
    }
  };

  // Step 3 of the grouping by layer: copies the selected rechits so that the ones of layer slot s fill
  // output[layerOffsets[s] .. layerOffsets[s] + layerCounts[s] - 1].
  // nWrittenInSlot[s] (zeroed by the caller) counts the rechits of slot s already written.
  struct HGCalRecHitCalibrationKernel_copyRecHitsByLayer {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalSoARecHitsDeviceCollection::View output,
                                  HGCalSoARecHitsDeviceCollection::ConstView recHits,
                                  int32_t const* __restrict__ sidx,
                                  uint32_t const* __restrict__ layerOffsets,
                                  uint32_t* __restrict__ nWrittenInSlot) const {
      for (auto isel : uniform_elements(acc, output.metadata().size())) {
        auto const& recHit = recHits[sidx[isel]];
        uint32_t const slot = hgcalrechitlayers::slot(recHit.layer(), recHit.dim3());
        uint32_t const positionInLayer = alpaka::atomicAdd(acc, &nWrittenInSlot[slot], 1u);
        output[layerOffsets[slot] + positionInLayer] = recHit;
      }
    }
  };

  // @short apply all calibration kernels
  HGCalSoARecHitsDeviceCollection HGCalRecHitCalibrationAlgorithms::calibrate(
      Queue& queue,
      int32_t* __restrict__ nsel,
      int32_t* __restrict__ sidx,
      uint32_t* __restrict__ layerCounts,
      HGCalDigiDevice const& device_digis,
      HGCalCalibParamDevice const& device_calib,
      HGCalMappingModuleParamDevice const& device_mapmod,
      HGCalMappingCellParamDevice const& device_cell_mapping,
      HGCalDenseIndexInfoDevice const& device_index,
      HGCalCMCorrectionDeviceCollection const& device_corrections,
      bool cmAltAsNominal,
      double k_noise) const {
    LogDebug("HGCalRecHitCalibrationAlgorithms") << "\n\nINFO -- Start of calibrate\n\n" << std::endl;
    auto const ndigis = device_digis.view().metadata().size();

    LogDebug("HGCalRecHitCalibrationAlgorithms")
        << "\n\nINFO -- Allocating rechits buffer and initiating values" << std::endl;
    HGCalSoARecHitsDeviceCollection device_recHits(queue, ndigis);

    // number of items per group
    uint32_t items = n_threads_;
    // use as many groups as needed to cover the whole problem
    uint32_t groups = divide_up_by(ndigis, items);
    // map items to
    //   - threads with a single element per thread on a GPU backend
    //   - elements within a single thread on a CPU backend
    auto grid = make_workdiv<Acc1D>(groups, items);
    LogDebug("HGCalRecHitCalibrationAlgorithms") << "N groups: " << groups << "\tN items: " << items << std::endl;

    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalRecHitCalibrationKernel_flagRecHits{},
                        device_recHits.view(),
                        device_digis.const_view(),
                        device_calib.const_view());
    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalRecHitCalibrationKernel_adcToEnergy{cmAltAsNominal},
                        device_recHits.view(),
                        device_digis.const_view(),
                        device_calib.const_view(),
                        device_corrections.const_view());
    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalRecHitCalibrationKernel_toaToTime{timeResolution_},
                        device_recHits.view(),
                        device_digis.const_view(),
                        device_calib.const_view());
    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalRecHitCalibrationKernel_handleCalibCell{timeResolution_},
                        device_recHits.view(),
                        device_digis.const_view(),
                        device_calib.const_view(),
                        device_cell_mapping.const_view(),
                        device_index.const_view());
    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalRecHitCalibrationKernel_metaData{},
                        device_recHits.view(),
                        device_mapmod.const_view(),
                        device_index.const_view());

    // select rechits and count them per layer slot (step 1 of the grouping by layer)
    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalRecHitCalibrationKernel_selectRecHits{},
                        nsel,
                        sidx,
                        layerCounts,
                        device_recHits.const_view(),
                        device_cell_mapping.const_view(),
                        device_index.const_view(),
                        k_noise);

    return device_recHits;
  }

  // @short select rechits
  HGCalSoARecHitsDeviceCollection HGCalRecHitCalibrationAlgorithms::select(
      Queue& queue,
      int const ndigis,
      int32_t const* __restrict__ nsel,
      int32_t const* __restrict__ sidx,
      uint32_t const* __restrict__ layerOffsets,
      HGCalSoARecHitsDeviceCollection const& device_recHits) const {
    // number of items per group
    uint32_t items = n_threads_;
    // use as many groups as needed to cover the whole problem
    uint32_t groups = divide_up_by(ndigis, items);
    // map items to
    //   - threads with a single element per thread on a GPU backend
    //   - elements within a single thread on a CPU backend
    auto grid = make_workdiv<Acc1D>(groups, items);

    HGCalSoARecHitsDeviceCollection device_selRecHits(queue, *nsel);
    // the number of rechits already written in each layer slot
    auto nWrittenInSlot = make_device_buffer<uint32_t[]>(queue, hgcalrechitlayers::kLayerSlots);
    alpaka::memset(queue, nWrittenInSlot, 0x0);
    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalRecHitCalibrationKernel_copyRecHitsByLayer{},
                        device_selRecHits.view(),
                        device_recHits.const_view(),
                        sidx,
                        layerOffsets,
                        nWrittenInSlot.data());

    LogDebug("HGCalRecHitCalibrationAlgorithms") << "Input recHits: " << std::endl;
#ifdef EDM_ML_DEBUG
    int n_hits_to_print = 10;
    print_recHit_device(queue, *device_selRecHits, n_hits_to_print);
#endif

    return device_selRecHits;
  }

  void HGCalRecHitCalibrationAlgorithms::print(HGCalDigiHost const& digis, int max) const {
    int max_ = max > 0 ? max : digis.view().metadata().size();
    for (int i = 0; i < max_; i++) {
      LogDebug("HGCalRecHitCalibrationAlgorithms")
          << i << digis.view()[i].tot() << "\t" << digis.view()[i].toa() << "\t" << digis.view()[i].cm() << "\t"
          << digis.view()[i].flags() << std::endl;
    }
  }

  void HGCalRecHitCalibrationAlgorithms::print_digi_device(HGCalDigiDevice const& digis, int max) const {
    int max_ = max > 0 ? max : digis.view().metadata().size();
    for (int i = 0; i < max_; i++) {
      LogDebug("HGCalRecHitCalibrationAlgorithms")
          << i << digis.view()[i].tot() << "\t" << digis.view()[i].toa() << "\t" << digis.view()[i].cm() << "\t"
          << digis.view()[i].flags() << std::endl;
    }
  }

  void HGCalRecHitCalibrationAlgorithms::print_recHit_device(Queue& queue,
                                                             HGCalSoARecHitsHostCollection::View const& recHits,
                                                             int max) const {
    auto grid = make_workdiv<Acc1D>(1, 1);
    auto size = max > 0 ? max : recHits.metadata().size();
    alpaka::exec<Acc1D>(queue, grid, HGCalRecHitCalibrationKernel_printRecHits{}, recHits, size);

    // ensure that the print operations are complete before returning
    alpaka::wait(queue);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
