#ifndef DataFormats_HGCalReco_interface_HGCalSoARecHits_h
#define DataFormats_HGCalReco_interface_HGCalSoARecHits_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/HGCalReco/interface/HGCalRecoVariations.h"

// SoA layout with dim1, dim2, weight, sigmaNoise, recHitsIndex layer and cellsCount fields
GENERATE_SOA_LAYOUT(HGCalSoARecHitsLayout,
                    // columns: one value per element
                    SOA_COLUMN(float, dim1),
                    SOA_COLUMN(float, dim2),
                    SOA_COLUMN(float, dim3),
                    SOA_COLUMN(int, layer),
                    SOA_COLUMN(float, energy),
                    SOA_EIGEN_COLUMN(hgcalreco::VariationVector, energy_variations),  // by hgcalreco::Variation
                    SOA_COLUMN(float, mipEnergy),
                    SOA_COLUMN(float, sigmaNoise),
                    SOA_COLUMN(unsigned int, recHitIndex),
                    SOA_COLUMN(uint16_t, flags),
                    SOA_COLUMN(uint32_t, detid),
                    SOA_COLUMN(float, time),       // ns, -99 if none
                    SOA_COLUMN(float, timeError),  // ns, -1 if no time
                    // by hgcalreco::Variation: time and time error with the energy of the variation (same rechits
                    // with a time as the nominal one)
                    SOA_EIGEN_COLUMN(hgcalreco::VariationVector, time_variations),
                    SOA_EIGEN_COLUMN(hgcalreco::VariationVector, timeError_variations))

using HGCalSoARecHits = HGCalSoARecHitsLayout<>;

enum HGCalRecHitFlags { kNormal = 0x0, kEnergyInvalid = 0x1, kTimeInvalid = 0x2 };

#endif  // DataFormats_PortableTestObjects_interface_TestSoA_h
