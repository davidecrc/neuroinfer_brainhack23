// Function to update the Papaya viewer with or without a mask
window.update_papaya_viewer = function(mask = false) {
    // Setting the images for the Papaya viewer based on the presence of a mask
    if (mask) {
        params["images"] = ["../templates/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz", "../.tmp/mask.nii.gz"];
    } else {
        params["images"] = ["../templates/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz"];
    }

    // Getting the visualizer element and creating a new Papaya viewer
    var visualizer = document.getElementById("visualizer");
    papaya.Container.addViewer("visualizer", params);
}
