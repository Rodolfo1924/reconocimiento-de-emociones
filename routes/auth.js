const express = require("express");
const router = express.Router();

// Ruta para login
router.post("/login", (req, res) => {
  // Lógica de autenticación
  res.json({ message: "Login exitoso" });
});

// Ruta para registro
router.post("/register", (req, res) => {
  // Lógica de registro
  res.status(201).json({ message: "Registro exitoso" });
});

module.exports = router;
