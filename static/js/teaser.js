class TeaserController {
    constructor() {
        // Get DOM elements
        this.mainView = document.querySelector('.main-view');
        this.videoView = document.querySelector('.video-view');
        this.zoomInVideo = document.getElementById('zoom-in-video');
        this.interactionVideo = document.getElementById('interaction-video');
        this.zoomOutVideo = document.getElementById('zoom-out-video');
        this.backButton = document.querySelector('.back-button');
        
        // Define video paths
        this.regions = {
            region1: {
                zoomIn: 'assets/teaser/honey-zoom-in.mp4',
                interaction: 'assets/teaser/honey.mp4',
                zoomOut: 'assets/teaser/honey-zoom-out.mp4'
            },
            region2: {
                zoomIn: 'assets/teaser/hat-zoom-in.mp4',
                interaction: 'assets/teaser/hat.mp4',
                zoomOut: 'assets/teaser/hat-zoom-out.mp4'
            },
            region3: {
                zoomIn: 'assets/teaser/breeze-zoom-in.mp4',
                interaction: 'assets/teaser/breeze.mp4',
                zoomOut: 'assets/teaser/breeze-zoom-out.mp4'
            },
            region4: {
                zoomIn: 'assets/teaser/glass-zoom-in.mp4',
                interaction: 'assets/teaser/glass.mp4',
                zoomOut: 'assets/teaser/glass-zoom-out.mp4'
            }
        };

        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Set up hotspot clicks
        document.querySelectorAll('.hotspot').forEach(hotspot => {
            hotspot.addEventListener('click', (e) => {
                this.playRegionSequence(e.currentTarget.dataset.region);
            });
        });

        // Set up back button
        this.backButton.addEventListener('click', () => this.returnToMain());

        // Set up video endings
        this.zoomInVideo.addEventListener('ended', () => this.startInteraction());
        this.zoomOutVideo.addEventListener('ended', () => this.showMainView());
    }

    playRegionSequence(region) {
        const videos = this.regions[region];
        
        // Set up videos
        this.zoomInVideo.src = videos.zoomIn;
        this.interactionVideo.src = videos.interaction;
        this.zoomOutVideo.src = videos.zoomOut;

        // Switch views
        this.mainView.style.visibility = 'hidden';
        this.videoView.style.visibility = 'visible';
        this.zoomInVideo.style.visibility = 'visible';
        this.zoomInVideo.play();
    }

    startInteraction() {
        this.zoomInVideo.style.visibility = 'hidden';
        this.interactionVideo.style.visibility = 'visible';
        this.backButton.style.visibility = 'visible';
        this.interactionVideo.play();
    }

    returnToMain() {
        this.interactionVideo.style.visibility = 'hidden';
        this.backButton.style.visibility = 'hidden';
        this.zoomOutVideo.style.visibility = 'visible';
        this.zoomOutVideo.play();
    }

    showMainView() {
        this.videoView.style.visibility = 'hidden';
        this.zoomOutVideo.style.visibility = 'hidden';
        this.mainView.style.visibility = 'visible';
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new TeaserController();
});