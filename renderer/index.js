// renderer/index.js

window.addEventListener('DOMContentLoaded', async () => {
  try {
    // Pedir las emociones desde el proceso principal a través del preload
    const emociones = await window.emocionesAPI.leerEmociones();

    // Contar ocurrencias de cada emoción
    const conteo = {};
    emociones.forEach(item => {
      conteo[item.emocion] = (conteo[item.emocion] || 0) + 1;
    });

    // Preparar etiquetas y datos para la gráfica
    const etiquetas = Object.keys(conteo);
    const datos = Object.values(conteo);

    // Obtener contexto del canvas
    const ctx = document.getElementById('graficaEmociones').getContext('2d');

    // Crear la gráfica de barras con Chart.js
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: etiquetas,
        datasets: [{
          label: 'Número de veces detectada',
          data: datos,
          backgroundColor: 'rgba(54, 162, 235, 0.7)',
          borderColor: 'rgba(54, 162, 235, 1)',
          borderWidth: 1
        }]
      },
      options: {
        scales: {
          y: {
            beginAtZero: true,
            stepSize: 1
          }
        }
      }
    });

  } catch (error) {
    console.error('Error al cargar o mostrar emociones:', error);
  }
});
