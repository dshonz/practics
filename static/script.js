// Функция для мобильного меню
document.addEventListener('DOMContentLoaded', function() {
    // Мобильное меню (адаптивность)
    const mobileMenuButton = document.createElement('button');
    mobileMenuButton.className = 'mobile-menu-button';
    mobileMenuButton.innerHTML = '☰ Меню';
    document.querySelector('header .container').appendChild(mobileMenuButton);
    
    const nav = document.querySelector('nav ul');
    mobileMenuButton.addEventListener('click', function() {
        nav.style.display = nav.style.display === 'flex' ? 'none' : 'flex';
    });

    // Адаптация меню при изменении размера экрана
    function handleResize() {
        if (window.innerWidth > 768) {
            nav.style.display = 'flex';
        } else {
            nav.style.display = 'none';
        }
    }

    window.addEventListener('resize', handleResize);
    handleResize();

    // Плавная прокрутка для якорей
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Анимация карточек услуг при прокрутке
    const serviceCards = document.querySelectorAll('.service-card');
    const animateOnScroll = function() {
        serviceCards.forEach(card => {
            const cardPosition = card.getBoundingClientRect().top;
            const screenPosition = window.innerHeight / 1.3;

            if (cardPosition < screenPosition) {
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }
        });
    };

    // Инициализация анимации
    serviceCards.forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'all 0.5s ease';
    });

    window.addEventListener('scroll', animateOnScroll);
    animateOnScroll(); // Запустить при загрузке

    // Валидация форм
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            let valid = true;
            const inputs = this.querySelectorAll('input[required]');
            
            inputs.forEach(input => {
                if (!input.value.trim()) {
                    input.style.borderColor = 'red';
                    valid = false;
                } else {
                    input.style.borderColor = '';
                }
            });

            if (!valid) {
                e.preventDefault();
                alert('Пожалуйста, заполните все обязательные поля');
            }
        });
    });

    // Маска для телефона
    const phoneInputs = document.querySelectorAll('input[type="tel"]');
    phoneInputs.forEach(input => {
        input.addEventListener('input', function() {
            this.value = this.value.replace(/[^\d+]/g, '');
            
            // Простая маска для российских номеров
            if (this.value.startsWith('+7') && this.value.length > 2) {
                let value = this.value.replace(/\D/g, '');
                let formattedValue = '+7';
                
                if (value.length > 1) {
                    formattedValue += ' (' + value.substring(1, 4);
                }
                if (value.length > 4) {
                    formattedValue += ') ' + value.substring(4, 7);
                }
                if (value.length > 7) {
                    formattedValue += '-' + value.substring(7, 9);
                }
                if (value.length > 9) {
                    formattedValue += '-' + value.substring(9, 11);
                }
                
                this.value = formattedValue;
            }
        });
    });
});

// Функция для отображения текущего года в футере
function updateYear() {
    const yearElement = document.querySelector('footer p:first-child');
    if (yearElement) {
        const currentYear = new Date().getFullYear();
        yearElement.textContent = yearElement.textContent.replace(/\d{4}/, currentYear);
    }
}

// Инициализация при загрузке
document.addEventListener('DOMContentLoaded', updateYear);