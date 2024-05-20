// Function to update the Papaya viewer with or without a mask
window.update_papaya_viewer = function(mask = false) {
    params["coordinate"]=currentCoords;
    if (Array.isArray(mask)) {
        // If mask is an array, read the first element
        overlays = mask;
        current_overlay = overlays[0];
    } else if (typeof mask === 'string') {
        // If mask is a string, keep it as a string
        current_overlay = mask;
    } else if (mask) {
        current_overlay = "/.tmp/mask.nii.gz";
    }
    if (!mask) {
        // If mask is neither an array nor a string, set the images without a mask
        params["images"] = ["/templates/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz"];
        current_overlay = "/templates/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz";
    }
    else {
        params["images"] = ["/templates/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz",  current_overlay];
    }

    // Getting the visualizer element and creating a new Papaya viewer
    papaya.Container.resetViewer(0, params);
}
