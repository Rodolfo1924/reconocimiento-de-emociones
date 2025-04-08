const { connectToServer } = require('./connection');

(async () => {
  try {
    console.log("🔌 Iniciando prueba de conexión...");
    const db = await connectToServer();
    
    // Verificación más completa
    console.log("\n🔍 Estadísticas de la base de datos:");
    const stats = await db.stats();
    console.log("- Nombre DB:", stats.db);
    console.log("- Colecciones:", stats.collections);
    console.log("- Tamaño:", (stats.dataSize / 1024 / 1024).toFixed(2), "MB");
    
    // Listar colecciones con manejo de vacío
    const collections = await db.listCollections().toArray();
    console.log("\n📂 Colecciones disponibles:");
    if (collections.length === 0) {
      console.log("(No hay colecciones)");
    } else {
      console.log(collections.map(c => `- ${c.name}`).join('\n'));
    }
    
  } catch (err) {
    console.error("\n❌ Falla en la prueba:", err.message);
  } finally {
    console.log("\n🏁 Prueba finalizada");
    process.exit();
  }
})();