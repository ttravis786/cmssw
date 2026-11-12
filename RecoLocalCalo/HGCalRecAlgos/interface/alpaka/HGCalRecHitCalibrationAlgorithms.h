#ifndef RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalRecHitCalibrationAlgorithms_h
#define RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalRecHitCalibrationAlgorithms_h

#include <cmath>
#include <cstdint>

// Alpaka imports
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// HGCal digis data formats
#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
#include "DataFormats/HGCalDigi/interface/HGCROCChannelDataFrame.h"

// Host & devide HGCal RecHit data formats
#include "DataFormats/HGCalDigi/interface/HGCalDigiHost.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalDigiDevice.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsHostCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsDeviceCollection.h"
#include "CondFormats/HGCalObjects/interface/HGCalCalibrationParameterHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalCalibrationParameterDevice.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalMappingParameterDevice.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalCMCorrectionDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace hgcaldigi;
  using namespace hgcalrechit;
  using namespace hgcal;

  // Time resolution of a rechit, in ns, from its signal over noise x = energy / sigmaNoise: sqrt((a / x)^2 + c^2),
  // with x below minSoN set to minSoN and c above maxSoN. Same as ComputeClusterTime::getTimeError("recHit").
  struct HGCalRecHitTimeResolution {
    static constexpr float kNoTime = -99.f;      // time of a rechit without a time
    static constexpr float kNoTimeError = -1.f;  // its time error

    float noiseTerm;  // a, in ns
    float constTerm;  // c, in ns
    float minSoN;
    float maxSoN;

    ALPAKA_FN_HOST_ACC float operator()(float energy, float sigmaNoise) const {
      if (sigmaNoise <= 0.f) {
        return kNoTimeError;
      }
      float const son = energy / sigmaNoise;
      if (son > maxSoN) {
        return constTerm;
      }
      float const a = noiseTerm / (son < minSoN ? minSoN : son);
      return std::sqrt(a * a + constTerm * constTerm);
    }
  };

  // Grouping the selected rechits by layer
  // --------------------------------------
  // The layer clustering (CLUEstering) clusters all layers in one go, each layer on its own. It needs the rechits of
  // each layer stored next to each other, plus the number of rechits in each layer (the "layerSizes" product).
  // The digis are in readout order (FED, ECON-D), which is not layer order, and the zero suppression keeps a different
  // set of rechits in every event, so the layer boundaries are worked out per event, as a counting sort in 3 steps:
  //
  //   1. calibrate(), kernel _selectRecHits: for every rechit that passes the selection, add 1 to the counter of its
  //      layer slot (layerCounts) and store its index in sidx.
  //   2. HGCalRecHitsProducer::produce(), on the host: layerOffsets[s] = sum of layerCounts[0..s-1], i.e. where layer
  //      slot s starts in the output. The non-empty counts become layerSizes.
  //   3. select(), kernel _copyRecHitsByLayer: each selected rechit is written at layerOffsets[slot] + n, where n is
  //      how many rechits of that slot were already written (an atomic counter per slot).
  //
  //   Example, 6 selected rechits with layers 3 1 3 2 1 3:
  //     layerCounts  : slot1 = 2, slot2 = 1, slot3 = 3
  //     layerOffsets : slot1 = 0, slot2 = 2, slot3 = 3
  //     output       : [1 1 | 2 | 3 3 3],  layerSizes = {2, 1, 3}
  //
  // The GPU threads run in no fixed order, so within one layer the order of the rechits is arbitrary. The clustering
  // does not depend on it.
  //
  // A layer slot combines the layer number and the endcap: layer l (0 <= l < kMaxLayersPerSide) at z >= 0 is slot l,
  // at z < 0 it is slot l + kMaxLayersPerSide. Slot kLayerSlots collects rechits whose layer is out of range (an error,
  // checked in produce()).
  namespace hgcalrechitlayers {
    constexpr uint32_t kMaxLayersPerSide = 64;
    constexpr uint32_t kLayerSlots = 2 * kMaxLayersPerSide;

    // the layer slot of a rechit, from its layer number and its z position (dim3)
    ALPAKA_FN_HOST_ACC inline uint32_t slot(int layer, float z) {
      if (layer < 0 || layer >= static_cast<int>(kMaxLayersPerSide)) {
        return kLayerSlots;
      }
      return static_cast<uint32_t>(layer) + (z < 0.f ? kMaxLayersPerSide : 0u);
    }
  }  // namespace hgcalrechitlayers

  class HGCalRecHitCalibrationAlgorithms {
  public:
    HGCalRecHitCalibrationAlgorithms(int n_blocks, int n_threads, HGCalRecHitTimeResolution const& timeResolution)
        : n_blocks_(n_blocks), n_threads_(n_threads), timeResolution_(timeResolution) {}

    // energy_variations(CM_ALT) uses device_corrections where valid, else equals energy;
    // cmAltAsNominal swaps energy and energy_variations(CM_ALT).
    // Also does step 1 of the grouping by layer (see hgcalrechitlayers above): writes the number of selected rechits
    // in nsel, their indices in sidx, and their number per layer slot in layerCounts (kLayerSlots + 1 entries, zeroed
    // by the caller).
    HGCalSoARecHitsDeviceCollection calibrate(
        Queue& queue,
        int32_t* __restrict__ nsel,
        int32_t* __restrict__ sidx,
        uint32_t* __restrict__ layerCounts,
        HGCalDigiDevice const& device_digis,
        HGCalCalibParamDevice const& device_calib,
        HGCalMappingModuleParamDevice const& device_mapmod,
        HGCalMappingCellParamDevice const& device_mapping,
        HGCalDenseIndexInfoDevice const& device_index,
        HGCalCMCorrectionDeviceCollection const&
            device_corrections,  // SOA has two columns is valid and the correction itself.
        bool cmAltAsNominal,  // This swaps energy and energy variations i.e. if I want the ML method to be the default.
        double k_noise) const;

    // Step 3 of the grouping by layer: copies the selected rechits so that the ones of layer slot s start at
    // layerOffsets[s] (a device pointer)
    HGCalSoARecHitsDeviceCollection select(Queue& queue,
                                           int const ndigis,
                                           int32_t const* __restrict__ nsel,
                                           int32_t const* __restrict__ sidx,
                                           uint32_t const* __restrict__ layerOffsets,
                                           HGCalSoARecHitsDeviceCollection const& device_recHits) const;

  private:
    void print(HGCalDigiHost const& digis, int max = -1) const;
    void print_digi_device(HGCalDigiDevice const& digis, int max = -1) const;
    void print_recHit_device(Queue& queue, HGCalSoARecHitsHostCollection::View const& recHits, int max = -1) const;

    int n_blocks_;
    int n_threads_;
    HGCalRecHitTimeResolution timeResolution_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalRecHitCalibrationAlgorithms_h
