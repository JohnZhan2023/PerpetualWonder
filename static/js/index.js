window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function () {
  // Check for click events on the navbar burger icon
  var options = {
    slidesToScroll: 1,
    slidesToShow: 1,
    loop: true,
    infinite: true,
    autoplay: true,
    autoplaySpeed: 5000,
  };

  // Initialize all div with carousel class
  var carousels = bulmaCarousel.attach(".carousel", options);
  bulmaSlider.attach();
});

// Wait until the DOM is fully loaded (optional but recommended)
document.addEventListener("DOMContentLoaded", () => {
  // Define text button content for each scene
  const sceneTextButtons = {
    0: [ // Jam scene
      "Pour the jam onto the cake",
      "Pour while moving & shaking",
      "Pour while moving left & right",
      "Pour much more jam"
    ],
    1: [ // Smoke scene
      "Emit smoke",
      "Breeze mildly towards left",
      "Breeze strongly towards right",
      "Breeze strongly towards left"
    ],
    2: [ // Clothes scene
      "Breeze towards left",
      "Breeze left & right",
      "Breeze in mixed ways", 
      "Breeze in mixed ways"
    ],
    3: [ // Sand scene
      "Pull downward",
      "Pull towards right",
      "Pull towards left",
      "Make the sand castle thinner"
    ]
  };

  // Function to update text buttons
  function updateTextButtons(sceneIndex) {
    const textButtons = document.querySelectorAll('#gallery-buttons-diff-actions .button p');
    const texts = sceneTextButtons[sceneIndex];
    textButtons.forEach((button, i) => {
      button.textContent = texts[i];
    });
  }

  // Add click event listeners to scene buttons
  const sceneButtons = document.querySelectorAll('#gallery-buttons-diff-actions-scenes .scene-button');
  sceneButtons.forEach((button, index) => {
    button.addEventListener('click', () => {
      // Update text buttons
      updateTextButtons(index);
      
      // Update scene button states
      sceneButtons.forEach(btn => btn.classList.remove('is-primary'));
      button.classList.add('is-primary');
      
      // Reset action buttons - make first one active
      const actionButtons = document.querySelectorAll('#gallery-buttons-diff-actions .button');
      actionButtons.forEach((btn, i) => {
        btn.classList.toggle('is-primary', i === 0);
      });

      // Calculate video index (first video of the new scene)
      const videoIndex = index * 4;
      
      // Switch to the corresponding video
      const videoCarousel = document.getElementById('gallery-carousel-diff-actions');
      const videoItems = videoCarousel.querySelectorAll('.video-gallery-item');
      
      videoItems.forEach((item, i) => {
        if (i === videoIndex) {
          item.classList.add('active');
          const video = item.querySelector('video');
          if (video) {
            video.currentTime = 0;
            video.play().catch(e => console.log("Video play prevented:", e));
          }
        } else {
          item.classList.remove('active');
          const video = item.querySelector('video');
          if (video) {
            video.pause();
            video.currentTime = 0;
          }
        }
      });
    });
  });

  // Add click event listeners to action buttons
  const actionButtons = document.querySelectorAll('#gallery-buttons-diff-actions .button');
  actionButtons.forEach((button, actionIndex) => {
    button.addEventListener('click', () => {
      // Update action button states
      actionButtons.forEach(btn => btn.classList.remove('is-primary'));
      button.classList.add('is-primary');

      // Find current scene index
      const activeSceneButton = document.querySelector('#gallery-buttons-diff-actions-scenes .scene-button.is-primary');
      const sceneIndex = Array.from(sceneButtons).indexOf(activeSceneButton);

      // Calculate video index
      const videoIndex = (sceneIndex * 4) + actionIndex;

      // Switch to the corresponding video
      const videoCarousel = document.getElementById('gallery-carousel-diff-actions');
      const videoItems = videoCarousel.querySelectorAll('.video-gallery-item');
      
      videoItems.forEach((item, i) => {
        if (i === videoIndex) {
          item.classList.add('active');
          const video = item.querySelector('video');
          if (video) {
            video.currentTime = 0;
            video.play().catch(e => console.log("Video play prevented:", e));
          }
        } else {
          item.classList.remove('active');
          const video = item.querySelector('video');
          if (video) {
            video.pause();
            video.currentTime = 0;
          }
        }
      });
    });
  });

  // Original toggle functionality
  const toggleHeading = document.getElementById("toggle-resim");
  const content = document.getElementById("resim-content");

  // Make sure these elements exist before attaching event listeners
  if (toggleHeading && content) {
    toggleHeading.addEventListener("click", () => {
      // Toggle the 'display' style
      if (content.style.display === "none" || content.style.display === "") {
        content.style.display = "block";
      } else {
        content.style.display = "none";
      }
    });
  }

  // ====== For Ablation Novel View Synthesis ======
  const toggleAblNvs = document.getElementById("toggle-abl-nvs");
  const ablNvsContent = document.getElementById("abl-nvs-content");
  if (toggleAblNvs && ablNvsContent) {
    toggleAblNvs.addEventListener("click", () => {
      // Toggle the display
      if (ablNvsContent.style.display === "none" || ablNvsContent.style.display === "") {
        ablNvsContent.style.display = "block";
      } else {
        ablNvsContent.style.display = "none";
      }
    });
  }

  // ====== For Ablation Re-simulation ======
  const toggleAblResim = document.getElementById("toggle-abl-resim");
  const ablResimContent = document.getElementById("abl-resim-content");
  if (toggleAblResim && ablResimContent) {
    toggleAblResim.addEventListener("click", () => {
      // Toggle the display
      if (ablResimContent.style.display === "none" || ablResimContent.style.display === "") {
        ablResimContent.style.display = "block";
      } else {
        ablResimContent.style.display = "none";
      }
    });
  }
});