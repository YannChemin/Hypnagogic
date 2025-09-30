#ifndef MATERIAL_COLORS_H
#define MATERIAL_COLORS_H

#include <gdal.h>

// Color palette for material classification
// Organized by material category with distinct color schemes
// Total: 287 predefined colors matching MaterialType enum

static const GDALColorEntry MATERIAL_COLORS[] = {
    // Vegetation - Various greens by growth stage/type (0-28)
    {34, 139, 34, 255},       // 0: VEGETATION - Forest Green
    {0, 128, 0, 255},         // 1: CORN_LATE_VEGETATIVE - Green
    {85, 107, 47, 255},       // 2: CORN_TASSELING - Dark Olive Green
    {107, 142, 35, 255},      // 3: CORN_GRAIN_FILLING - Olive Drab
    {128, 128, 0, 255},       // 4: CORN_MATURITY - Olive
    {0, 255, 0, 255},         // 5: CORN_IRRIGATED - Bright Green
    {154, 205, 50, 255},      // 6: CORN_STRESSED - Yellow Green
    
    // Wheat - Golden/brown tones (7-13)
    {173, 255, 47, 255},      // 7: WHEAT_TILLERING - Green Yellow
    {127, 255, 0, 255},       // 8: WHEAT_JOINTING - Chartreuse
    {255, 215, 0, 255},       // 9: WHEAT_HEADING - Gold
    {218, 165, 32, 255},      // 10: WHEAT_GRAIN_FILLING - Golden Rod
    {184, 134, 11, 255},      // 11: WHEAT_SENESCENCE - Dark Golden Rod
    {255, 255, 0, 255},       // 12: WHEAT_IRRIGATED - Yellow
    {189, 183, 107, 255},     // 13: WHEAT_STRESSED - Dark Khaki
    
    // Soybeans - Green variations (14-19)
    {46, 125, 50, 255},       // 14: SOYBEAN_VEGETATIVE - Dark Green
    {76, 175, 80, 255},       // 15: SOYBEAN_FLOWERING - Green
    {139, 195, 74, 255},      // 16: SOYBEAN_POD_FILLING - Light Green
    {205, 220, 57, 255},      // 17: SOYBEAN_MATURITY - Lime
    {102, 187, 106, 255},     // 18: SOYBEAN_IRRIGATED - Light Green
    {165, 214, 167, 255},     // 19: SOYBEAN_STRESSED - Very Light Green
    
    // Rice - Blue-green tones (20-24)
    {0, 128, 128, 255},       // 20: RICE_VEGETATIVE - Teal
    {64, 224, 208, 255},      // 21: RICE_REPRODUCTIVE - Turquoise
    {72, 209, 204, 255},      // 22: RICE_RIPENING - Medium Turquoise
    {0, 206, 209, 255},       // 23: RICE_FLOODED - Dark Turquoise
    {95, 158, 160, 255},      // 24: RICE_UPLAND - Cadet Blue
    
    // Cotton - White/cream tones (25-30)
    {255, 228, 225, 255},     // 25: COTTON_SQUARING - Misty Rose
    {255, 182, 193, 255},     // 26: COTTON_FLOWERING - Light Pink
    {255, 255, 240, 255},     // 27: COTTON_BOLL_DEVELOPMENT - Ivory
    {245, 245, 220, 255},     // 28: COTTON_DEFOLIATION - Beige
    {255, 255, 255, 255},     // 29: COTTON_IRRIGATED - White
    {220, 220, 220, 255},     // 30: COTTON_STRESSED - Gainsboro
    
    // Other Field Crops - Earth tones (31-49)
    {210, 180, 140, 255},     // 31: BARLEY_TILLERING - Tan
    {222, 184, 135, 255},     // 32: BARLEY_STEM_ELONGATION - Burlywood
    {205, 133, 63, 255},      // 33: BARLEY_HEADING - Peru
    {160, 82, 45, 255},       // 34: BARLEY_GRAIN_FILLING - Saddle Brown
    
    {238, 203, 173, 255},     // 35: OATS_VEGETATIVE - Peach Puff
    {255, 218, 185, 255},     // 36: OATS_HEADING - Peach Puff
    {255, 222, 173, 255},     // 37: OATS_GRAIN_FILLING - Navajo White
    
    {255, 140, 0, 255},       // 38: SUNFLOWER_VEGETATIVE - Dark Orange
    {255, 165, 0, 255},       // 39: SUNFLOWER_BUD_FORMATION - Orange
    {255, 215, 0, 255},       // 40: SUNFLOWER_FLOWERING - Gold
    {184, 134, 11, 255},      // 41: SUNFLOWER_SEED_FILLING - Dark Golden Rod
    
    {255, 255, 0, 255},       // 42: CANOLA_ROSETTE - Yellow
    {255, 215, 0, 255},       // 43: CANOLA_STEM_ELONGATION - Gold
    {255, 255, 0, 255},       // 44: CANOLA_FLOWERING - Yellow
    {218, 165, 32, 255},      // 45: CANOLA_POD_FILLING - Golden Rod
    
    {128, 0, 128, 255},       // 46: SUGAR_BEET_EARLY - Purple
    {138, 43, 226, 255},      // 47: SUGAR_BEET_CANOPY_CLOSURE - Blue Violet
    {75, 0, 130, 255},        // 48: SUGAR_BEET_ROOT_GROWTH - Indigo
    {75, 0, 130, 255},        // 49: SUGAR_BEET_NEUSLING - Indigo
    
    {143, 188, 143, 255},     // 50: ALFALFA_EARLY_VEGETATIVE - Dark Sea Green
    {60, 179, 113, 255},      // 51: ALFALFA_BUD_STAGE - Medium Sea Green
    {152, 251, 152, 255},     // 52: ALFALFA_FLOWERING - Pale Green
    {32, 178, 170, 255},      // 53: ALFALFA_POST_CUT - Light Sea Green
    
    // Water (54)
    {0, 0, 255, 255},         // 54: WATER - Blue
    
    // Igneous Rocks - Dark grays/blacks (55-61)
    {105, 105, 105, 255},     // 55: GRANITE - Dim Gray
    {47, 79, 79, 255},        // 56: BASALT - Dark Slate Gray
    {112, 128, 144, 255},     // 57: GABBRO - Slate Gray
    {0, 0, 0, 255},           // 58: OBSIDIAN - Black
    {220, 220, 220, 255},     // 59: PUMICE - Gainsboro
    {169, 169, 169, 255},     // 60: RHYOLITE - Dark Gray
    {128, 128, 128, 255},     // 61: ANDESITE - Gray
    
    // Sedimentary Rocks - Light grays/tans (62-68)
    {245, 245, 220, 255},     // 62: LIMESTONE - Beige
    {238, 203, 173, 255},     // 63: SANDSTONE - Peach Puff
    {119, 136, 153, 255},     // 64: SHALE - Light Slate Gray
    {160, 82, 45, 255},       // 65: MUDSTONE - Saddle Brown
    {205, 192, 176, 255},     // 66: CONGLOMERATE - Light Gray
    {255, 248, 220, 255},     // 67: DOLOMITE - Cornsilk
    {255, 255, 255, 255},     // 68: GYPSUM - White
    
    // Metamorphic Rocks - Medium grays (69-73)
    {255, 228, 225, 255},     // 69: MARBLE - Misty Rose
    {240, 248, 255, 255},     // 70: QUARTZITE - Alice Blue
    {112, 128, 144, 255},     // 71: SLATE - Slate Gray
    {169, 169, 169, 255},     // 72: SCHIST - Dark Gray
    {128, 128, 128, 255},     // 73: GNEISS - Gray
    
    // Individual Minerals (74-83)
    {255, 255, 255, 255},     // 74: QUARTZ - White
    {255, 20, 147, 255},      // 75: FELDSPAR - Deep Pink
    {255, 255, 224, 255},     // 76: CALCITE - Light Yellow
    {255, 248, 220, 255},     // 77: DOLOMITE_MINERAL - Cornsilk
    {255, 255, 255, 255},     // 78: KAOLINITE - White
    {169, 169, 169, 255},     // 79: MONTMORILLONITE - Dark Gray
    {128, 128, 128, 255},     // 80: ILLITE - Gray
    {205, 92, 92, 255},       // 81: HEMATITE - Indian Red
    {255, 140, 0, 255},       // 82: GOETHITE - Dark Orange
    {0, 0, 0, 255},           // 83: MAGNETITE - Black
    
    // Soils and Sediments (84-88)
    {255, 218, 185, 255},     // 84: SAND - Peach Puff
    {160, 82, 45, 255},       // 85: CLAY - Saddle Brown
    {139, 69, 19, 255},       // 86: SOIL - Saddle Brown
    {178, 34, 34, 255},       // 87: LATERITE - Fire Brick
    {85, 107, 47, 255},       // 88: PEAT - Dark Olive Green
    
    // Hydrocarbons - Blacks and dark colors (89-107)
    {47, 79, 79, 255},        // 89: CRUDE_OIL_LIGHT - Dark Slate Gray
    {25, 25, 25, 255},        // 90: CRUDE_OIL_MEDIUM - Very Dark Gray
    {0, 0, 0, 255},           // 91: CRUDE_OIL_HEAVY - Black
    {0, 0, 0, 255},           // 92: CRUDE_OIL_EXTRA_HEAVY - Black
    
    {255, 20, 147, 255},      // 93: GASOLINE - Deep Pink
    {255, 140, 0, 255},       // 94: DIESEL - Dark Orange
    {135, 206, 235, 255},     // 95: JET_FUEL - Sky Blue
    {255, 69, 0, 255},        // 96: HEATING_OIL - Orange Red
    {139, 69, 19, 255},       // 97: LUBRICATING_OIL - Saddle Brown
    {0, 0, 0, 255},           // 98: ASPHALT - Black
    
    {173, 216, 230, 255},     // 99: PROPANE - Light Blue
    {255, 182, 193, 255},     // 100: BUTANE - Light Pink
    {240, 248, 255, 255},     // 101: NATURAL_GAS_CONDENSATE - Alice Blue
    
    {72, 61, 139, 255},       // 102: WEATHERED_OIL_LIGHT - Dark Slate Blue
    {47, 79, 79, 255},        // 103: WEATHERED_OIL_MEDIUM - Dark Slate Gray
    {25, 25, 25, 255},        // 104: WEATHERED_OIL_HEAVY - Very Dark Gray
    {139, 69, 19, 255},       // 105: OIL_MOUSSE - Saddle Brown
    
    // Plastics - Bright synthetic colors (106-119)
    {255, 0, 255, 255},       // 106: POLYETHYLENE_LD - Magenta
    {255, 20, 147, 255},      // 107: POLYETHYLENE_HD - Deep Pink
    {138, 43, 226, 255},      // 108: POLYPROPYLENE - Blue Violet
    {255, 105, 180, 255},     // 109: POLYSTYRENE - Hot Pink
    {148, 0, 211, 255},       // 110: PVC - Dark Violet
    {0, 255, 255, 255},       // 111: PET - Cyan
    {30, 144, 255, 255},      // 112: POLYCARBONATE - Dodger Blue
    {255, 0, 0, 255},         // 113: NYLON - Red
    {255, 69, 0, 255},        // 114: ABS - Orange Red
    {255, 182, 193, 255},     // 115: PMMA - Light Pink
    {240, 248, 255, 255},     // 116: PTFE - Alice Blue
    
    {218, 112, 214, 255},     // 117: PLASTIC_WEATHERED_UV - Orchid
    {221, 160, 221, 255},     // 118: PLASTIC_WEATHERED_THERMAL - Plum
    {255, 192, 203, 255},     // 119: MICROPLASTICS - Pink
    
    // Man-made Materials (120-123)
    {192, 192, 192, 255},     // 120: CONCRETE - Silver
    {105, 105, 105, 255},     // 121: METAL - Dim Gray
    {255, 127, 80, 255},      // 122: PAINT - Coral
    {255, 255, 255, 255},     // 123: SNOW - White
    
    // Roofing Materials (124-152)
    {25, 25, 25, 255},        // 124: ASPHALT_SHINGLES - Very Dark Gray
    {205, 92, 92, 255},       // 125: CLAY_TILES - Indian Red
    {169, 169, 169, 255},     // 126: CONCRETE_TILES - Dark Gray
    {112, 128, 144, 255},     // 127: SLATE_ROOFING - Slate Gray
    {210, 180, 140, 255},     // 128: WOOD_SHINGLES - Tan
    {192, 192, 192, 255},     // 129: METAL_ROOFING_STEEL - Silver
    {211, 211, 211, 255},     // 130: METAL_ROOFING_ALUMINUM - Light Gray
    {184, 115, 51, 255},      // 131: METAL_ROOFING_COPPER - Dark Orange
    {105, 105, 105, 255},     // 132: METAL_ROOFING_ZINC - Dim Gray
    {128, 128, 128, 255},     // 133: METAL_ROOFING_TIN - Gray
    
    // Flat/Commercial Roofing (134-143)
    {47, 79, 79, 255},        // 134: BUILT_UP_ROOFING - Dark Slate Gray
    {0, 0, 0, 255},           // 135: MODIFIED_BITUMEN - Black
    {25, 25, 25, 255},        // 136: EPDM_RUBBER - Very Dark Gray
    {255, 255, 255, 255},     // 137: TPO_MEMBRANE - White
    {240, 248, 255, 255},     // 138: PVC_MEMBRANE - Alice Blue
    {192, 192, 192, 255},     // 139: LIQUID_APPLIED_MEMBRANE - Silver
    {34, 139, 34, 255},       // 140: GREEN_ROOF_EXTENSIVE - Forest Green
    {0, 128, 0, 255},         // 141: GREEN_ROOF_INTENSIVE - Green
    {255, 255, 255, 255},     // 142: WHITE_MEMBRANE - White
    {128, 128, 128, 255},     // 143: BALLASTED_ROOF - Gray
    
    // Temporary/Emergency Roofing (144-151)
    {0, 0, 255, 255},         // 144: TARP_BLUE - Blue
    {192, 192, 192, 255},     // 145: TARP_SILVER - Silver
    {255, 255, 255, 255},     // 146: TARP_WHITE - White
    {210, 180, 140, 255},     // 147: TARP_CANVAS - Tan
    {169, 169, 169, 255},     // 148: TEMPORARY_METAL_SHEET - Dark Gray
    {160, 82, 45, 255},       // 149: PLYWOOD_SHEETING - Saddle Brown
    {240, 248, 255, 255},     // 150: PLASTIC_SHEETING - Alice Blue
    {255, 69, 0, 255},        // 151: EMERGENCY_PATCH - Orange Red
    
    // Solar/Renewable Energy (152-161)
    {25, 25, 112, 255},       // 152: SOLAR_PV_MONOCRYSTALLINE - Midnight Blue
    {0, 0, 139, 255},         // 153: SOLAR_PV_POLYCRYSTALLINE - Dark Blue
    {47, 79, 79, 255},        // 154: SOLAR_PV_THIN_FILM_CDTE - Dark Slate Gray
    {72, 61, 139, 255},       // 155: SOLAR_PV_THIN_FILM_CIGS - Dark Slate Blue
    {105, 105, 105, 255},     // 156: SOLAR_PV_THIN_FILM_AMORPHOUS - Dim Gray
    {0, 0, 205, 255},         // 157: SOLAR_PV_BIFACIAL - Medium Blue
    {65, 105, 225, 255},      // 158: SOLAR_THERMAL_FLAT - Royal Blue
    {100, 149, 237, 255},     // 159: SOLAR_THERMAL_EVACUATED - Cornflower Blue
    {211, 211, 211, 255},     // 160: PV_MOUNTING_ALUMINUM - Light Gray
    {105, 105, 105, 255},     // 161: PV_MOUNTING_STEEL - Dim Gray
    
    // Specialty Roofing (162-171)
    {218, 165, 32, 255},      // 162: THATCH_ROOFING - Golden Rod
    {107, 142, 35, 255},      // 163: SOD_ROOF - Olive Drab
    {255, 255, 255, 255},     // 164: MEMBRANE_REFLECTIVE - White
    {0, 0, 139, 255},         // 165: PHOTOVOLTAIC_INTEGRATED - Dark Blue
    {173, 216, 230, 255},     // 166: SKYLIGHTS_GLASS - Light Blue
    {240, 248, 255, 255},     // 167: SKYLIGHTS_PLASTIC - Alice Blue
    {255, 255, 255, 255},     // 168: ROOF_COATING_ELASTOMERIC - White
    {248, 248, 255, 255},     // 169: ROOF_COATING_SILICONE - Ghost White
    {255, 255, 255, 255},     // 170: ROOF_COATING_ACRYLIC - White
    
    // Marine/Harbor Materials - Boat Decks (171-182)
    {245, 245, 220, 255},     // 171: FIBERGLASS_DECK - Beige
    {210, 180, 140, 255},     // 172: WOOD_DECK_TEAK - Tan
    {160, 82, 45, 255},       // 173: WOOD_DECK_MAHOGANY - Saddle Brown
    {222, 184, 135, 255},     // 174: WOOD_DECK_PINE - Burlywood
    {192, 192, 192, 255},     // 175: ALUMINUM_DECK - Silver
    {128, 128, 128, 255},     // 176: STEEL_DECK_PAINTED - Gray
    {169, 169, 169, 255},     // 177: STEEL_DECK_GALVANIZED - Dark Gray
    {139, 69, 19, 255},       // 178: COMPOSITE_DECK - Saddle Brown
    {0, 0, 0, 255},           // 179: RUBBER_DECK_MATTING - Black
    {255, 255, 255, 255},     // 180: VINYL_DECK_COVERING - White
    {210, 180, 140, 255},     // 181: CANVAS_DECK_COVERING - Tan
    {255, 140, 0, 255},       // 182: ANTI_SLIP_COATING - Dark Orange
    
    // Hull and Structural (183-193)
    {255, 255, 255, 255},     // 183: FIBERGLASS_HULL_WHITE - White
    {30, 144, 255, 255},      // 184: FIBERGLASS_HULL_COLORED - Dodger Blue
    {192, 192, 192, 255},     // 185: ALUMINUM_HULL - Silver
    {105, 105, 105, 255},     // 186: STEEL_HULL_PAINTED - Dim Gray
    {160, 82, 45, 255},       // 187: WOOD_HULL_VARNISHED - Saddle Brown
    {255, 140, 0, 255},       // 188: INFLATABLE_PVC - Dark Orange
    {0, 0, 0, 255},           // 189: INFLATABLE_HYPALON - Black
    {255, 255, 255, 255},     // 190: MARINE_VINYL - White
    {210, 180, 140, 255},     // 191: BOAT_CANVAS - Tan
    {255, 0, 0, 255},         // 192: MARINE_PAINT_ANTIFOUL - Red
    {255, 255, 255, 255},     // 193: MARINE_PAINT_TOPSIDE - White
    
    // Harbor/Marina Infrastructure (194-205)
    {169, 169, 169, 255},     // 194: CONCRETE_DOCK - Dark Gray
    {160, 82, 45, 255},       // 195: WOOD_DOCK_TREATED - Saddle Brown
    {139, 69, 19, 255},       // 196: WOOD_DOCK_COMPOSITE - Saddle Brown
    {192, 192, 192, 255},     // 197: ALUMINUM_DOCK - Silver
    {255, 255, 255, 255},     // 198: PLASTIC_DOCK_HDPE - White
    {169, 169, 169, 255},     // 199: DOCK_DECKING_EZDOCK - Dark Gray
    {128, 128, 128, 255},     // 200: MARINA_PILING_CONCRETE - Gray
    {105, 105, 105, 255},     // 201: MARINA_PILING_STEEL - Dim Gray
    {160, 82, 45, 255},       // 202: MARINA_PILING_WOOD - Saddle Brown
    {255, 255, 255, 255},     // 203: FLOATING_DOCK_FOAM - White
    {192, 192, 192, 255},     // 204: BOAT_LIFT_ALUMINUM - Silver
    {0, 0, 0, 255},           // 205: FENDER_SYSTEM_RUBBER - Black
    
    // Water and Marine Environment (206-211)
    {0, 191, 255, 255},       // 206: WATER_HARBOR_CLEAN - Deep Sky Blue
    {139, 69, 19, 255},       // 207: WATER_HARBOR_TURBID - Saddle Brown
    {0, 0, 0, 255},           // 208: WATER_HARBOR_POLLUTED - Black
    {46, 125, 50, 255},       // 209: SEAWEED_FLOATING - Dark Green
    {245, 245, 220, 255},     // 210: BARNACLE_FOULING - Beige
    {0, 128, 0, 255},         // 211: MARINE_ALGAE - Green
    
    // Road Materials - Asphalt Types (212-217)
    {47, 79, 79, 255},        // 212: ASPHALT_NEW - Dark Slate Gray
    {105, 105, 105, 255},     // 213: ASPHALT_AGED - Dim Gray
    {169, 169, 169, 255},     // 214: ASPHALT_CRACKED - Dark Gray
    {128, 128, 128, 255},     // 215: ASPHALT_RECYCLED - Gray
    {220, 20, 60, 255},       // 216: ASPHALT_COLORED - Crimson
    {112, 128, 144, 255},     // 217: ASPHALT_POROUS - Slate Gray
    
    // Concrete Types (218-223)
    {211, 211, 211, 255},     // 218: CONCRETE_PAVEMENT_NEW - Light Gray
    {169, 169, 169, 255},     // 219: CONCRETE_PAVEMENT_AGED - Dark Gray
    {128, 128, 128, 255},     // 220: CONCRETE_PAVEMENT_STAINED - Gray
    {192, 192, 192, 255},     // 221: CONCRETE_STAMPED - Silver
    {169, 169, 169, 255},     // 222: CONCRETE_EXPOSED_AGGREGATE - Dark Gray
    {255, 182, 193, 255},     // 223: CONCRETE_COLORED - Light Pink
    
    // Alternative Surfaces (224-231)
    {205, 92, 92, 255},       // 224: BRICK_PAVEMENT - Indian Red
    {105, 105, 105, 255},     // 225: STONE_PAVEMENT_GRANITE - Dim Gray
    {47, 79, 79, 255},        // 226: STONE_PAVEMENT_BASALT - Dark Slate Gray
    {112, 128, 144, 255},     // 227: COBBLESTONE - Slate Gray
    {205, 192, 176, 255},     // 228: GRAVEL_ROAD - Light Gray
    {160, 82, 45, 255},       // 229: DIRT_ROAD_DRY - Saddle Brown
    {139, 69, 19, 255},       // 230: DIRT_ROAD_WET - Saddle Brown
    {128, 128, 128, 255},     // 231: CRUSHED_STONE - Gray
    
    // Road Markings and Accessories (232-237)
    {255, 255, 255, 255},     // 232: ROAD_PAINT_WHITE - White
    {255, 255, 0, 255},       // 233: ROAD_PAINT_YELLOW - Yellow
    {255, 255, 255, 255},     // 234: THERMOPLASTIC_MARKING - White
    {255, 255, 255, 255},     // 235: REFLECTIVE_ROAD_MARKING - White
    {0, 0, 0, 255},           // 236: ROAD_SEALANT - Black
    {47, 79, 79, 255},        // 237: POTHOLE_PATCH - Dark Slate Gray
    
    // Road Infrastructure (238-242)
    {169, 169, 169, 255},     // 238: CONCRETE_CURB - Dark Gray
    {192, 192, 192, 255},     // 239: STEEL_GUARDRAIL - Silver
    {169, 169, 169, 255},     // 240: CONCRETE_BARRIER - Dark Gray
    {105, 105, 105, 255},     // 241: ASPHALT_SHOULDER - Dim Gray
    {128, 128, 128, 255},     // 242: GRAVEL_SHOULDER - Gray
    
    // Parking and Urban Surfaces (243-248)
    {47, 79, 79, 255},        // 243: PARKING_LOT_ASPHALT - Dark Slate Gray
    {169, 169, 169, 255},     // 244: PARKING_LOT_CONCRETE - Dark Gray
    {192, 192, 192, 255},     // 245: INTERLOCKING_PAVERS - Silver
    {211, 211, 211, 255},     // 246: PERMEABLE_PAVERS - Light Gray
    {255, 140, 0, 255},       // 247: RUBBER_SPEED_BUMP - Dark Orange
    {255, 255, 255, 255},     // 248: PAINTED_CROSSWALK - White
    
    // Forest and Tree Species - Deciduous Broadleaf (249-263)
    {34, 139, 34, 255},       // 249: OAK_TREE - Forest Green
    {255, 0, 0, 255},         // 250: MAPLE_TREE - Red (fall colors)
    {255, 255, 255, 255},     // 251: BIRCH_TREE - White (bark)
    {160, 82, 45, 255},       // 252: BEECH_TREE - Saddle Brown
    {128, 128, 128, 255},     // 253: ASH_TREE - Gray
    {139, 69, 19, 255},       // 254: HICKORY_TREE - Saddle Brown
    {255, 182, 193, 255},     // 255: CHERRY_TREE - Light Pink
    {139, 69, 19, 255},       // 256: WALNUT_TREE - Saddle Brown
    {255, 255, 255, 255},     // 257: POPLAR_TREE - White
    {255, 255, 0, 255},       // 258: WILLOW_TREE - Yellow (weeping)
    {210, 180, 140, 255},     // 259: BASSWOOD_TREE - Tan
    {160, 82, 45, 255},       // 260: CHESTNUT_TREE - Saddle Brown
    {245, 245, 220, 255},     // 261: SYCAMORE_TREE - Beige (bark pattern)
    {255, 140, 0, 255},       // 262: TULIP_TREE - Dark Orange (flowers)
    {255, 0, 0, 255},         // 263: SWEETGUM_TREE - Red (fall colors)
    
    // Evergreen Coniferous - Various greens (264-275)
    {0, 100, 0, 255},         // 264: PINE_TREE - Dark Green
    {34, 139, 34, 255},       // 265: SPRUCE_TREE - Forest Green
    {0, 128, 0, 255},         // 266: FIR_TREE - Green
    {85, 107, 47, 255},       // 267: CEDAR_TREE - Dark Olive Green
    {46, 125, 50, 255},       // 268: HEMLOCK_TREE - Dark Green
    {107, 142, 35, 255},      // 269: DOUGLAS_FIR - Olive Drab
    {128, 128, 0, 255},       // 270: JUNIPER_TREE - Olive
    {173, 255, 47, 255},      // 271: LARCH_TREE - Green Yellow (deciduous conifer)
    {165, 42, 42, 255},       // 272: REDWOOD_TREE - Brown (bark)
    {160, 82, 45, 255},       // 273: SEQUOIA_TREE - Saddle Brown (bark)
    {46, 125, 50, 255},       // 274: CYPRESS_TREE - Dark Green
    {0, 100, 0, 255},         // 275: YEW_TREE - Dark Green
    
    // Tropical and Subtropical Trees (276-287)
    {0, 128, 0, 255},         // 276: PALM_TREE - Green
    {255, 255, 0, 255},       // 277: BANANA_TREE - Yellow
    {0, 128, 0, 255},         // 278: MANGO_TREE - Green
    {0, 100, 0, 255},         // 279: AVOCADO_TREE - Dark Green
    {255, 165, 0, 255},       // 280: CITRUS_TREE - Orange
    {139, 69, 19, 255},       // 281: COCONUT_PALM - Saddle Brown
    {160, 82, 45, 255},       // 282: BAOBAB_TREE - Saddle Brown
    {139, 69, 19, 255},       // 283: MAHOGANY_TREE - Saddle Brown
    {210, 180, 140, 255},     // 284: TEAK_TREE - Tan
    {0, 0, 0, 255},           // 285: EBONY_TREE - Black
    {245, 245, 220, 255},     // 286: BALSA_TREE - Beige
    {165, 42, 42, 255},       // 287: BRAZILWOOD_TREE - Brown
};

// Number of predefined colors
#define NUM_MATERIAL_COLORS (sizeof(MATERIAL_COLORS) / sizeof(GDALColorEntry))

#endif // MATERIAL_COLORS_H
