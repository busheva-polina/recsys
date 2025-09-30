// Global variables
let model;
let isTraining = false;

// Initialize when window loads
window.onload = async function() {
    console.log('Initializing application...');
    updateTrainingStatus('Loading data...', 'loading');
    
    try {
        // Load data
        await loadData();
        updateTrainingStatus('Data loaded successfully!', 'success');
        
        // Populate dropdowns
        populateUserDropdown();
        populateMovieDropdown();
        
        // Start training
        await trainModel();
        
    } catch (error) {
        console.error('Initialization error:', error);
        updateTrainingStatus('Error during initialization: ' + error.message, 'error');
    }
};

// Update training status in UI
function updateTrainingStatus(message, type = 'loading') {
    const statusElement = document.getElementById('training-status');
    statusElement.textContent = message;
    statusElement.className = type;
}

// Populate user dropdown
function populateUserDropdown() {
    const userSelect = document.getElementById('user-select');
    userSelect.innerHTML = '';
    
    // Create options for users (using first 100 users for demo)
    const maxUsers = Math.min(numUsers, 100);
    for (let i = 1; i < maxUsers; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `User ${i}`;
        userSelect.appendChild(option);
    }
}

// Populate movie dropdown
function populateMovieDropdown() {
    const movieSelect = document.getElementById('movie-select');
    movieSelect.innerHTML = '';
    
    // Use available movies from data
    movies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        movieSelect.appendChild(option);
    });
}

// Model Definition Function
function createModel(numUsers, numMovies, latentDim = 10) {
    console.log(`Creating model with ${numUsers} users, ${numMovies} movies, latent dimension: ${latentDim}`);
    
    // Input Layers
    const userInput = tf.input({ shape: [1], name: 'userInput' });
    const movieInput = tf.input({ shape: [1], name: 'movieInput' });
    
    // Embedding Layers
    const userEmbedding = tf.layers.embedding({
        inputDim: numUsers,
        outputDim: latentDim,
        name: 'userEmbedding'
    }).apply(userInput);
    
    const movieEmbedding = tf.layers.embedding({
        inputDim: numMovies,
        outputDim: latentDim,
        name: 'movieEmbedding'
    }).apply(movieInput);
    
    // Reshape embeddings to remove the sequence dimension
    const userVector = tf.layers.flatten().apply(userEmbedding);
    const movieVector = tf.layers.flatten().apply(movieEmbedding);
    
    // Dot product of user and movie vectors
    const dotProduct = tf.layers.dot({ axes: 1 }).apply([userVector, movieVector]);
    
    // Add bias terms
    const userBias = tf.layers.embedding({
        inputDim: numUsers,
        outputDim: 1,
        name: 'userBias'
    }).apply(userInput);
    
    const movieBias = tf.layers.embedding({
        inputDim: numMovies,
        outputDim: 1,
        name: 'movieBias'
    }).apply(movieInput);
    
    const flattenedUserBias = tf.layers.flatten().apply(userBias);
    const flattenedMovieBias = tf.layers.flatten().apply(movieBias);
    
    // Combine dot product with biases
    const prediction = tf.layers.add().apply([
        dotProduct,
        flattenedUserBias,
        flattenedMovieBias
    ]);
    
    // Scale to rating range (1-5)
    const scaledPrediction = tf.layers.dense({
        units: 1,
        activation: 'sigmoid',
        kernelInitializer: 'zeros',
        biasInitializer: tf.initializers.constant({ value: 3.0 })
    }).apply(prediction);
    
    const finalPrediction = tf.layers.multiply().apply([
        tf.layers.add().apply([
            scaledPrediction,
            tf.layers.lambda({
                function: (x) => tf.mul(x, 0) // Create zero tensor with same shape
            }).apply(scaledPrediction)
        ]),
        tf.layers.lambda({
            function: (x) => tf.add(tf.mul(x, 0), 4) // Create constant 4 tensor
        }).apply(scaledPrediction)
    ]);
    
    const ratingPrediction = tf.layers.add().apply([
        finalPrediction,
        tf.layers.lambda({
            function: (x) => tf.add(tf.mul(x, 0), 1) // Create constant 1 tensor
        }).apply(finalPrediction)
    ]);
    
    // Create model
    const model = tf.model({
        inputs: [userInput, movieInput],
        outputs: ratingPrediction,
        name: 'MatrixFactorization'
    });
    
    return model;
}

// Training Function
async function trainModel() {
    if (isTraining) {
        console.log('Model is already training...');
        return;
    }
    
    isTraining = true;
    const predictBtn = document.getElementById('predict-btn');
    predictBtn.disabled = true;
    
    try {
        updateTrainingStatus('Creating model architecture...', 'loading');
        
        // Create model with smaller latent dimension for faster training
        model = createModel(numUsers, numMovies, 8);
        
        updateTrainingStatus('Compiling model...', 'loading');
        
        // Compile model
        model.compile({
            optimizer: tf.train.adam(0.01), // Higher learning rate for faster convergence
            loss: 'meanSquaredError',
            metrics: ['mse']
        });
        
        // Prepare training data
        updateTrainingStatus('Preparing training data...', 'loading');
        
        const userIDs = ratings.map(r => r.userId);
        const movieIDs = ratings.map(r => r.movieId);
        const ratingValues = ratings.map(r => r.rating);
        
        const userTensor = tf.tensor2d(userIDs, [userIDs.length, 1]);
        const movieTensor = tf.tensor2d(movieIDs, [movieIDs.length, 1]);
        const ratingTensor = tf.tensor2d(ratingValues, [ratingValues.length, 1]);
        
        updateTrainingStatus('Starting training... (This may take a moment)', 'loading');
        
        // Train model with fewer epochs for faster completion
        const history = await model.fit([userTensor, movieTensor], ratingTensor, {
            epochs: 8,
            batchSize: 32,
            validationSplit: 0.1,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    updateTrainingStatus(`Epoch ${epoch + 1}/8 - Loss: ${logs.loss.toFixed(4)}`, 'loading');
                    console.log(`Epoch ${epoch + 1}, Loss: ${logs.loss}`);
                }
            }
        });
        
        // Clean up tensors
        userTensor.dispose();
        movieTensor.dispose();
        ratingTensor.dispose();
        
        updateTrainingStatus('✅ Model training completed successfully! Ready for predictions.', 'success');
        predictBtn.disabled = false;
        
        console.log('Training completed. Final loss:', history.history.loss[history.history.loss.length - 1]);
        
    } catch (error) {
        console.error('Training error:', error);
        updateTrainingStatus('❌ Error during training: ' + error.message, 'error');
    } finally {
        isTraining = false;
    }
}

// Prediction Function
async function predictRating() {
    const userSelect = document.getElementById('user-select');
    const movieSelect = document.getElementById('movie-select');
    const resultElement = document.getElementById('result');
    
    const userId = parseInt(userSelect.value);
    const movieId = parseInt(movieSelect.value);
    
    if (!userId || !movieId) {
        resultElement.innerHTML = '<p style="color: #f56565;">Please select both a user and a movie.</p>';
        return;
    }
    
    if (!model) {
        resultElement.innerHTML = '<p style="color: #f56565;">Model is not ready yet. Please wait for training to complete.</p>';
        return;
    }
    
    try {
        resultElement.innerHTML = '<p>Calculating prediction...</p>';
        
        // Create input tensors
        const userTensor = tf.tensor2d([[userId]]);
        const movieTensor = tf.tensor2d([[movieId]]);
        
        // Make prediction
        const prediction = model.predict([userTensor, movieTensor]);
        const rating = await prediction.data();
        const predictedRating = Math.min(5, Math.max(0, rating[0])); // Clamp between 0-5
        
        // Clean up tensors
        userTensor.dispose();
        movieTensor.dispose();
        prediction.dispose();
        
        // Display result with stars
        const stars = '★'.repeat(Math.round(predictedRating)) + '☆'.repeat(5 - Math.round(predictedRating));
        
        resultElement.innerHTML = `
            <div class="rating-display">${predictedRating.toFixed(1)}</div>
            <div class="rating-stars">${stars}</div>
            <p style="margin-top: 10px; color: #4a5568;">
                Predicted rating for ${movieSelect.options[movieSelect.selectedIndex].text} 
                by ${userSelect.options[userSelect.selectedIndex].text}
            </p>
        `;
        
    } catch (error) {
        console.error('Prediction error:', error);
        resultElement.innerHTML = `<p style="color: #f56565;">Error making prediction: ${error.message}</p>`;
    }
}
