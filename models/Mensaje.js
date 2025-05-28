const mongoose = require("mongoose");

const mensajeSchema = new mongoose.Schema({
  texto: {
    type: String,
    required: true
  }
});

module.exports = mongoose.model("Mensaje", mensajeSchema);
