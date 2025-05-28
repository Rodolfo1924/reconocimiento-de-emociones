const express = require("express");
const cors = require("cors");
const mongoose = require("mongoose");

// 1. Crear la app express primero
const app = express();

// 2. Middlewares
app.use(cors());
app.use(express.json());

// 3. Rutas
const authRoutes = require("./routes/auth");
const mensajeRouter = require("./routes/mensaje");

app.use("/api/auth", authRoutes);
app.use("/api/mensajes", mensajeRouter);

// 4. Conexión a MongoDB Atlas
mongoose.connect(process.env.MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
.then(() => console.log("✅ Conectado a MongoDB Atlas"))
.catch((err) => console.error("❌ Error conectando a MongoDB", err));

// 5. Servidor escuchando en puerto
const PORT = process.env.PORT || 10000;
app.listen(PORT, () => {
  console.log(`Servidor escuchando en puerto ${PORT}`);
});
