const express = require("express");
const router = express.Router();
const Mensaje = require("../models/Mensaje");

// Obtener todos los mensajes
router.get("/", async (req, res) => {
  const mensajes = await Mensaje.find();
  res.json(mensajes);
});

// Crear un nuevo mensaje
router.post("/", async (req, res) => {
  const nuevoMensaje = new Mensaje({ texto: req.body.texto });
  await nuevoMensaje.save();
  res.status(201).json(nuevoMensaje);
});

module.exports = router;
