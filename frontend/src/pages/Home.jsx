import "./styles.css";

const Home = ({ onStart }) => {
  const handleFeatureClick = (feature) => {
    console.log(`Clic en: ${feature}`);
  };

  return (
    <div className="container">
      <h1 className="title">
        Bien<span className="highlight">veni</span>do
      </h1>
      <p className="description">
        Con nuestra plataforma es muy fácil crear, compartir e interpretar
        emociones a partir de imágenes o videos.
      </p>
      <button onClick={onStart} className="button">
        Empezar
      </button>
      <div className="features-grid">
        {[
          "Reconocimiento Facial",
          "Carga de Archivos",
          "Estadísticas",
          "Emociones Detectadas",
          "Informes",
          "Soporte",
        ].map((item, index) => (
          <div
            key={index}
            onClick={() => handleFeatureClick(item)}
            role="button"
            tabIndex={0}
            onKeyPress={(e) => e.key === "Enter" && handleFeatureClick(item)}
            className="feature-item"
          >
            {item}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Home;