// Function to update the Papaya viewer with or without a mask
window.update_papaya_viewer = function(mask = false) {
    if (Array.isArray(mask)) {
        // If mask is an array, read the first element
        params["images"] = ["../templates/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz", "../" + mask[0]];
        console.log( "../" + mask[0])
    } else if (typeof mask === 'string') {
        // If mask is a string, keep it as a string
        params["images"] = ["../templates/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz",  "../" + mask];
        console.log( "../" + mask)
    } else {
        // If mask is neither an array nor a string, set the images without a mask
        params["images"] = ["../templates/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz",  "../.tmp/mask.nii.gz"];
    }

    // Getting the visualizer element and creating a new Papaya viewer
    var visualizer = document.getElementById("visualizer");
    papaya.Container.addViewer("visualizer", params);
}
