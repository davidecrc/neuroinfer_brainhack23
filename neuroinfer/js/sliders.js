function createGradient(maxHue) {
  const gradientColors = [];
  for (let i = 0; i <= maxHue; i += 1) {
    gradientColors.push(`hsl(${i}, 100%, 50%)`);
  }
  return `linear-gradient(to right, ${gradientColors.join(", ")})`;
}

function setSliderBackground(slider, maxHue) {
  slider.style.background = createGradient(maxHue);
}

function synchronizeSliders() {
  const slider1 = document.getElementById("colorRange1");
  const slider2 = document.getElementById("colorRange2");
  const slider3 = document.getElementById("colorRange3");

  slider1.addEventListener("input", () => {
    if (parseInt(slider1.value) >= parseInt(slider2.value)) {
      slider1.value = slider2.value - 1;
    }
  });

  slider2.addEventListener("input", () => {
    if (parseInt(slider2.value) <= parseInt(slider1.value)) {
      slider2.value = parseInt(slider1.value) + 1;
    }
    if (parseInt(slider2.value) >= parseInt(slider3.value)) {
      slider2.value = slider3.value - 1;
    }
  });

  slider3.addEventListener("input", () => {
    if (parseInt(slider3.value) <= parseInt(slider2.value)) {
      slider3.value = parseInt(slider2.value) + 1;
    }
  });
}

document
  .querySelectorAll(".slider")
  .forEach((slider) => setSliderBackground(slider, 275));
synchronizeSliders();
