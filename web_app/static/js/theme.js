// Theme Management
class ThemeManager {
    constructor() {
        this.theme = localStorage.getItem('theme') || 'dark';
        this.init();
    }

    init() {
        this.setTheme(this.theme);
        this.bindEvents();
    }

    setTheme(theme) {
        this.theme = theme;
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
        this.updateThemeButton();
    }

    toggleTheme() {
        const newTheme = this.theme === 'dark' ? 'light' : 'dark';
        this.setTheme(newTheme);
        this.showNotification(`Switched to ${newTheme} theme`, 'info');
    }

    updateThemeButton() {
        const button = document.getElementById('theme-toggle');
        if (button) {
            const icon = button.querySelector('i');
            if (icon) {
                icon.className = this.theme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
            }
        }
    }

    bindEvents() {
        const themeBtn = document.getElementById('theme-toggle');
        if (themeBtn) {
            themeBtn.addEventListener('click', () => this.toggleTheme());
        }

        // Listen for system theme changes
        if (window.matchMedia) {
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)');
            prefersDark.addEventListener('change', (e) => {
                if (!localStorage.getItem('theme')) {
                    this.setTheme(e.matches ? 'dark' : 'light');
                }
            });
        }
    }

    showNotification(message, type = 'info') {
        //
    }

    getIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }
}

// Initialize theme manager
function initTheme() {
    window.themeManager = new ThemeManager();
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ThemeManager, initTheme };
}