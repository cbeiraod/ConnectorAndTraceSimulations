"""
Central repository for PCB layer stackups.
Each stackup defines the thicknesses (mm) and electrical properties
of the layers from Top to Bottom.
"""

STACKUPS = {
    "2Layer_FR4": {
        "layers": [
            {"type": "copper", "thickness": 0.035, "name": "Top"},
            {"type": "dielectric", "thickness": 1.550, "epsilon_r": 4.4, "name": "Core"},
            {"type": "copper", "thickness": 0.035, "name": "Bottom"}
        ],
        # The index of the layer that acts as the reference ground
        "ref_gnd_idx": 2
    },
    "JLC_2Layer_FR4_1p6mm_1oz": {
        "layers": [
            {"type": "copper", "thickness": 0.035, "name": "Top"},
            {"type": "dielectric", "thickness": 1.50, "epsilon_r": 4.6, "name": "Core"},
            {"type": "copper", "thickness": 0.035, "name": "Bottom"}
        ],
        "ref_gnd_idx": 2
    },

    ## 4-Layer stackups
    "JLC_Generic4Layer_1p6mm_1oz_0p5oz": {
        "layers": [
            {"type": "copper", "thickness": 0.035, "name": "L1_Top"},
            {"type": "dielectric", "thickness": 0.2104, "epsilon_r": 4.4, "name": "7628 Prepreg"},
            {"type": "copper", "thickness": 0.0152, "name": "L2"},
            {"type": "dielectric", "thickness": 1.065, "epsilon_r": 4.6, "name": "Core"},
            {"type": "copper", "thickness": 0.0152, "name": "L3"},
            {"type": "dielectric", "thickness": 0.2104, "epsilon_r": 4.4, "name": "7628 Prepreg"},
            {"type": "copper", "thickness": 0.035, "name": "L4_Bottom"}
        ],
        "ref_gnd_idx": 2
    },
    "JLC04161H-7628": {
        "layers": [
            {"type": "copper", "thickness": 0.035, "name": "L1_Top"},
            {"type": "dielectric", "thickness": 0.2104, "epsilon_r": 4.4, "name": "7628 Prepreg"},
            {"type": "copper", "thickness": 0.0152, "name": "L2"},
            {"type": "dielectric", "thickness": 1.065, "epsilon_r": 4.6, "name": "Core"},
            {"type": "copper", "thickness": 0.0152, "name": "L3"},
            {"type": "dielectric", "thickness": 0.2104, "epsilon_r": 4.4, "name": "7628 Prepreg"},
            {"type": "copper", "thickness": 0.035, "name": "L4_Bottom"}
        ],
        "ref_gnd_idx": 2
    },
    "JLC04161H-3313": {
        "layers": [
            {"type": "copper", "thickness": 0.035, "name": "L1_Top"},
            {"type": "dielectric", "thickness": 0.0994, "epsilon_r": 4.1, "name": "3313 Prepreg"},
            {"type": "copper", "thickness": 0.0152, "name": "L2"},
            {"type": "dielectric", "thickness": 1.265, "epsilon_r": 4.6, "name": "Core"},
            {"type": "copper", "thickness": 0.0152, "name": "L3"},
            {"type": "dielectric", "thickness": 0.0994, "epsilon_r": 4.1, "name": "3313 Prepreg"},
            {"type": "copper", "thickness": 0.035, "name": "L4_Bottom"}
        ],
        "ref_gnd_idx": 2
    },
    "JLC04161H-1080": {
        "layers": [
            {"type": "copper", "thickness": 0.035, "name": "L1_Top"},
            {"type": "dielectric", "thickness": 0.0764, "epsilon_r": 3.91, "name": "1080 Prepreg"},
            {"type": "copper", "thickness": 0.0152, "name": "L2"},
            {"type": "dielectric", "thickness": 1.265, "epsilon_r": 4.6, "name": "Core"},
            {"type": "copper", "thickness": 0.0152, "name": "L3"},
            {"type": "dielectric", "thickness": 0.0764, "epsilon_r": 3.91, "name": "1080 Prepreg"},
            {"type": "copper", "thickness": 0.035, "name": "L4_Bottom"}
        ],
        "ref_gnd_idx": 2
    },
    "JLC04161H-2116": {
        "layers": [
            {"type": "copper", "thickness": 0.035, "name": "L1_Top"},
            {"type": "dielectric", "thickness": 0.1164, "epsilon_r": 4.16, "name": "2116 Prepreg"},
            {"type": "copper", "thickness": 0.0152, "name": "L2"},
            {"type": "dielectric", "thickness": 1.265, "epsilon_r": 4.6, "name": "Core"},
            {"type": "copper", "thickness": 0.0152, "name": "L3"},
            {"type": "dielectric", "thickness": 0.1164, "epsilon_r": 4.16, "name": "2116 Prepreg"},
            {"type": "copper", "thickness": 0.035, "name": "L4_Bottom"}
        ],
        "ref_gnd_idx": 2
    },

    ## 6-Layer stackups
    "JLC06161H-3313": {
        "layers": [
            {"type": "copper", "thickness": 0.035, "name": "L1_Top"},
            {"type": "dielectric", "thickness": 0.0994, "epsilon_r": 4.1, "name": "3313 Prepreg"},
            {"type": "copper", "thickness": 0.0152, "name": "L2"},
            {"type": "dielectric", "thickness": 0.55, "epsilon_r": 4.6, "name": "Core"},
            {"type": "copper", "thickness": 0.0152, "name": "L3"},
            {"type": "dielectric", "thickness": 0.1088, "epsilon_r": 4.16, "name": "2116 Prepreg"},
            {"type": "copper", "thickness": 0.0152, "name": "L4"},
            {"type": "dielectric", "thickness": 0.55, "epsilon_r": 4.6, "name": "Core"},
            {"type": "copper", "thickness": 0.0152, "name": "L5"},
            {"type": "dielectric", "thickness": 0.0994, "epsilon_r": 4.1, "name": "3313 Prepreg"},
            {"type": "copper", "thickness": 0.035, "name": "L6_Bottom"}
        ],
        "ref_gnd_idx": 2
    },
}