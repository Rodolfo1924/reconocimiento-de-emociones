const mongoose = require('mongoose');

const connectDB = async () => {
  try {
    await mongoose.connect('mongodb+srv://diegos:fljlGV3PxfuL4Wn7@reconocimiento.dfyanw8.mongodb.net/?retryWrites=true&w=majority&appName=reconocimiento');
    console.log('Conectado a MongoDB');
  } catch (error) {
    console.error('Error al conectar a MongoDB:', error);
    process.exit(1);
  }
};

module.exports = connectDB;