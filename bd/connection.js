const { MongoClient } = require('mongodb');

const uri = "mongodb://localhost:27017";
const dbName = "marketing_facial_db";

// Configuración simplificada (sin opciones obsoletas)
const client = new MongoClient(uri, {
  serverSelectionTimeoutMS: 3000 // Timeout de 3 segundos
});

let dbConnection;

async function connectToServer() {
  try {
    await client.connect();
    await client.db(dbName).command({ ping: 1 });
    console.log("✅ MongoDB conectado en:", uri);
    dbConnection = client.db(dbName);
    return dbConnection;
  } catch (err) {
    console.error("❌ Error de conexión:", err.message);
    throw err;
  }
}

function getDb() {
  if (!dbConnection) {
    throw new Error("No hay conexión a la base de datos");
  }
  return dbConnection;
}

module.exports = { connectToServer, getDb };