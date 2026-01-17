const VideoGallery = (function() {
    // Function to load a video lazily
    function loadVideo(video) {
        const source = video.querySelector('source[data-src]');
        if (source) {
            source.src = source.dataset.src;
            source.removeAttribute('data-src');
            video.load();
        }
    }

    // New Custom Slider Logic
    function initCustomSlider() {
        const track = document.getElementById('results-track');
        if (!track) return;
        
        // Find buttons within the container of the track
        const container = track.closest('.custom-slider-container');
        if (!container) return;

        const prevBtn = container.querySelector('.prev-btn');
        const nextBtn = container.querySelector('.next-btn');
        let columns = Array.from(track.querySelectorAll('.slider-column'));
        // columnWidth must match CSS: 260px
        const columnWidth = 260; 
        const visibleColumns = 3;
        const originalLength = columns.length; // 5
        
        // Clone first and last visibleColumns to create infinite loop effect
        // We clone visibleColumns amount to ensure we cover the view
        const clonesStart = columns.slice(-visibleColumns).map(col => col.cloneNode(true));
        const clonesEnd = columns.slice(0, visibleColumns).map(col => col.cloneNode(true));
        
        // Prepend and Append clones
        clonesStart.forEach(col => track.insertBefore(col, track.firstChild));
        clonesEnd.forEach(col => track.appendChild(col));
        
        // Initial index: We start after the prepended clones.
        // clonesStart length is visibleColumns (3).
        let currentIndex = visibleColumns;
        
        // Initial offset
        track.style.transform = `translateX(${-currentIndex * columnWidth}px)`;
        
        let isTransitioning = false;
        
        function updateSlider(withTransition = true) {
            if (withTransition) {
                track.style.transition = 'transform 0.4s ease';
                isTransitioning = true;
            } else {
                track.style.transition = 'none';
                isTransitioning = false;
            }
            const offset = -currentIndex * columnWidth;
            track.style.transform = `translateX(${offset}px)`;
        }
        
        track.addEventListener('transitionend', () => {
            isTransitioning = false;
            // Check for loop conditions
            // If we scrolled past the originals to the right (into clonesEnd)
            if (currentIndex >= originalLength + visibleColumns) {
                currentIndex = currentIndex - originalLength;
                updateSlider(false);
            }
            // If we scrolled past the originals to the left (into clonesStart)
            else if (currentIndex < visibleColumns) {
                currentIndex = currentIndex + originalLength;
                updateSlider(false);
            }
        });
        
        if (prevBtn) {
            prevBtn.addEventListener('click', () => {
                if (isTransitioning) return;
                currentIndex--;
                updateSlider(true);
            });
        }
        
        if (nextBtn) {
            nextBtn.addEventListener('click', () => {
                if (isTransitioning) return;
                currentIndex++;
                updateSlider(true);
            });
        }
        
        // Initialize
        updateSlider(false);
    }

    function handleVideoSwitch(galleryId, index, galleryType) {
        const container = document.getElementById(galleryId);
        if (!container) return;
        
        const items = container.querySelectorAll('.video-gallery-item');
        
        if (galleryType === 'results') {
            // For Results section: simple one-to-one mapping
            const track = document.getElementById('results-track');
            
            items.forEach((item, i) => {
                if (i === index) {
                    item.classList.add('active');
                    const video = item.querySelector('video');
                    if (video) {
                        loadVideo(video);
                        video.currentTime = 0;
                        video.muted = true;
                        video.play().catch(error => {
                            console.log("Autoplay was prevented:", error);
                        });
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

            // Update scene buttons - handle clones
            if (track) {
                const allButtons = track.querySelectorAll('.scene-button');
                allButtons.forEach(btn => {
                    btn.classList.remove('is-primary');
                    // Check against data-video-index since there are clones
                    if (parseInt(btn.dataset.videoIndex) === index) {
                        btn.classList.add('is-primary');
                    }
                });
            }
            
        } else if (galleryType === 'diff-actions') {
            // For Different Actions section
            const textButtons = container.parentElement.querySelector('#gallery-buttons-diff-actions').querySelectorAll('.button');
            const sceneButtons = container.parentElement.querySelector('#gallery-buttons-diff-actions-scenes').querySelectorAll('.scene-button');

            items.forEach((item, i) => {
                if (i === index) {
                    item.classList.add('active');
                    const video = item.querySelector('video');
                    if (video) {
                        loadVideo(video);
                        video.currentTime = 0;
                        video.muted = true;
                        video.play().catch(error => {
                            console.log("Autoplay was prevented:", error);
                        });
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

            textButtons.forEach(btn => btn.classList.remove('is-primary'));
            sceneButtons.forEach(btn => btn.classList.remove('is-primary'));
            
            const actionIndex = index % 4;
            const sceneIndex = Math.floor(index / 4);
            
            if (textButtons[actionIndex]) textButtons[actionIndex].classList.add('is-primary');
            if (sceneButtons[sceneIndex]) sceneButtons[sceneIndex].classList.add('is-primary');

        } else if (galleryType === 'comparisons') {
            // For Comparisons section
            const methodButtons = container.parentElement.querySelector('#gallery-buttons-nvs').querySelectorAll('.button');
            const sceneButtons = container.parentElement.querySelectorAll('.scene-button');

            items.forEach((item, i) => {
                if (i === index) {
                    item.classList.add('active');
                    const video = item.querySelector('video');
                    if (video) {
                        loadVideo(video);
                        video.currentTime = 0;
                        video.muted = true;
                        video.play().catch(error => {
                            console.log("Autoplay was prevented:", error);
                        });
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

            methodButtons.forEach(btn => btn.classList.remove('is-primary'));
            sceneButtons.forEach(btn => btn.classList.remove('is-primary'));
            
            const methodIndex = index % 3;
            const sceneIndex = Math.floor(index / 3);
            
            if (methodButtons[methodIndex]) methodButtons[methodIndex].classList.add('is-primary');
            if (sceneButtons[sceneIndex]) sceneButtons[sceneIndex].classList.add('is-primary');
        }
    }

    function initGallery(galleryButtonsId, galleryCarouselId, galleryType) {
        // Initialize carousel if it's the results gallery
        if (galleryType === 'results') {
             initCustomSlider();
        }

        const container = document.getElementById(galleryCarouselId);
        if (!container) return;

        // Only load the first video initially (the active one)
        const firstVideo = container.querySelector('.video-gallery-item.active video');
        if (firstVideo) {
            loadVideo(firstVideo);
            firstVideo.muted = true;
        }

        if (galleryType === 'results') {
            // Handle Results section using Event Delegation
            const track = document.getElementById('results-track');
            if (track) {
                track.addEventListener('click', (e) => {
                    const button = e.target.closest('.scene-button');
                    if (button) {
                        const index = parseInt(button.dataset.videoIndex);
                        if (!isNaN(index)) {
                            handleVideoSwitch(galleryCarouselId, index, 'results');
                        }
                    }
                });
            }
        } else if (galleryType === 'diff-actions') {
            // Handle Different Actions section
            const textButtons = document.getElementById('gallery-buttons-diff-actions').querySelectorAll('.button');
            const sceneButtons = document.getElementById('gallery-buttons-diff-actions-scenes').querySelectorAll('.scene-button');

            textButtons.forEach((button, actionIdx) => {
                button.addEventListener('click', () => {
                    const activeItem = container.querySelector('.video-gallery-item.active');
                    const currentIndex = Array.from(container.querySelectorAll('.video-gallery-item')).indexOf(activeItem);
                    const currentSceneIndex = Math.floor(currentIndex / 4);
                    const newIndex = currentSceneIndex * 4 + actionIdx;
                    handleVideoSwitch(galleryCarouselId, newIndex, 'diff-actions');
                });
            });

            sceneButtons.forEach((button, sceneIdx) => {
                button.addEventListener('click', () => {
                    const activeItem = container.querySelector('.video-gallery-item.active');
                    const currentIndex = Array.from(container.querySelectorAll('.video-gallery-item')).indexOf(activeItem);
                    const currentActionIndex = currentIndex % 4;
                    const newIndex = sceneIdx * 4 + currentActionIndex;
                    handleVideoSwitch(galleryCarouselId, newIndex, 'diff-actions');
                });
            });
        } else if (galleryType === 'comparisons') {
            // Handle Comparisons section
            const methodButtons = document.getElementById('gallery-buttons-nvs').querySelectorAll('.button');
            const sceneButtons = container.parentElement.querySelectorAll('.scene-button');

            methodButtons.forEach((button, methodIdx) => {
                button.addEventListener('click', () => {
                    const activeItem = container.querySelector('.video-gallery-item.active');
                    const currentIndex = Array.from(container.querySelectorAll('.video-gallery-item')).indexOf(activeItem);
                    const currentSceneIndex = Math.floor(currentIndex / 3);
                    const newIndex = currentSceneIndex * 3 + methodIdx;
                    handleVideoSwitch(galleryCarouselId, newIndex, 'comparisons');
                });
            });

            sceneButtons.forEach((button, sceneIdx) => {
                button.addEventListener('click', () => {
                    const activeItem = container.querySelector('.video-gallery-item.active');
                    const currentIndex = Array.from(container.querySelectorAll('.video-gallery-item')).indexOf(activeItem);
                    const currentMethodIndex = currentIndex % 3;
                    const newIndex = sceneIdx * 3 + currentMethodIndex;
                    handleVideoSwitch(galleryCarouselId, newIndex, 'comparisons');
                });
            });
        }

        // Play the first video
        if (firstVideo) {
            firstVideo.play().catch(error => {
                console.log("Initial autoplay was prevented:", error);
            });
        }
    }

    return {
        init: function() {
            // Initialize results gallery
            initGallery('gallery-buttons-results', 'gallery-carousel-results', 'results');
            
            // Initialize different actions gallery
            initGallery('gallery-buttons-diff-actions', 'gallery-carousel-diff-actions', 'diff-actions');
            
            // Initialize comparison gallery
            initGallery('gallery-buttons-nvs', 'gallery-carousel-nvs', 'comparisons');
        }
    };
})();

document.addEventListener('DOMContentLoaded', function() {
    VideoGallery.init();
});
