from visuomotor.utils.zarr_convertor import convert


if __name__ == "__main__":
    convert(
        "/home/andriisydor/masters_thesis/visuomotor_policy/data/IsaacLab_SD_08-08-2025-reviewed",
        "/home/andriisydor/masters_thesis/visuomotor_policy/data/SD_fixed_67_ep.zarr"
    )
