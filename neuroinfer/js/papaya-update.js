// Function to update the Papaya viewer with or without a mask
window.update_papaya_viewer = function (
  mask = false,
  minvalue = 0,
  maxvalue = 5,
) {
  params["coordinate"] = currentCoords;
  set_range = false;

  if (Array.isArray(mask)) {
    // If mask is an array, read the first element
    overlays = mask;
    current_overlay = overlays[0];
    overlayFilename = current_overlay.split("/").pop(); // Extract filename
    set_range = true;
  } else if (typeof mask === "string") {
    // If mask is a string, keep it as a string
    current_overlay = mask;
    overlayFilename = current_overlay.split("/").pop(); // Extract filename
    set_range = true;
  } else if (mask) {
    current_overlay = "/.tmp/mask.nii.gz";
  }

  if (!mask) {
    // If mask is neither an array nor a string, set the images without a mask
    params["images"] = [
      "/neuroinfer/templates/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz",
    ];
    current_overlay =
      "/neuroinfer/templates/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz";
  } else {
    params["images"] = [
      "/neuroinfer/templates/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz",
      current_overlay,
    ];
    if (set_range) {
      params[overlayFilename] = { min: minvalue, max: maxvalue };
    }
  }

  // Getting the visualizer element and creating a new Papaya viewer
  papaya.Container.resetViewer(0, params);
};
