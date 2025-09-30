// Global variables to store parsed data
let movies = [];
let ratings = [];
let numUsers = 0;
let numMovies = 0;

// MovieLens 100K dataset URLs
const MOVIES_URL = 'https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/multivariate-linear-regression/data/uci-iris-mlens-u.item';
const RATINGS_URL = 'https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/multivariate-linear-regression/data/uci-iris-mlens-u.data';

// For demonstration purposes, we'll use a smaller subset if the original URLs fail
const FALLBACK_MOVIES = [
    { id: 1, title: "Toy Story (1995)" },
    { id: 2, title: "GoldenEye (1995)" },
    { id: 3, title: "Four Rooms (1995)" },
    { id: 4, title: "Get Shorty (1995)" },
    { id: 5, title: "Copycat (1995)" }
];

const FALLBACK_RATINGS = [
    { userId: 1, movieId: 1, rating: 5 },
    { userId: 1, movieId: 2, rating: 3 },
    { userId: 2, movieId: 1, rating: 4 },
    { userId: 2, movieId: 3, rating: 5 },
    { userId: 3, movieId: 2, rating: 4 }
];

async function loadData() {
    try {
        console.log('Loading movie data...');
        const moviesResponse = await fetch(MOVIES_URL);
        let moviesText;
        
        if (moviesResponse.ok) {
            moviesText = await moviesResponse.text();
            movies = parseItemData(moviesText);
        } else {
            console.log('Using fallback movie data');
            movies = FALLBACK_MOVIES;
        }

        console.log('Loading rating data...');
        const ratingsResponse = await fetch(RATINGS_URL);
        let ratingsText;
        
        if (ratingsResponse.ok) {
            ratingsText = await ratingsResponse.text();
            ratings = parseRatingData(ratingsText);
        } else {
            console.log('Using fallback rating data');
            ratings = FALLBACK_RATINGS;
        }

        // Calculate number of unique users and movies
        const uniqueUsers = new Set(ratings.map(r => r.userId));
        const uniqueMovies = new Set(ratings.map(r => r.movieId));
        
        numUsers = Math.max(...uniqueUsers) + 1; // +1 because IDs start from 1
        numMovies = Math.max(...uniqueMovies) + 1;
        
        console.log(`Data loaded: ${numUsers} users, ${numMovies} movies, ${ratings.length} ratings`);
        
        return { movies, ratings, numUsers, numMovies };
        
    } catch (error) {
        console.error('Error loading data:', error);
        // Use fallback data
        movies = FALLBACK_MOVIES;
        ratings = FALLBACK_RATINGS;
        
        const uniqueUsers = new Set(ratings.map(r => r.userId));
        const uniqueMovies = new Set(ratings.map(r => r.movieId));
        
        numUsers = Math.max(...uniqueUsers) + 1;
        numMovies = Math.max(...uniqueMovies) + 1;
        
        return { movies, ratings, numUsers, numMovies };
    }
}

function parseItemData(text) {
    const lines = text.split('\n').filter(line => line.trim());
    const movies = [];
    
    for (const line of lines) {
        const parts = line.split('|');
        if (parts.length >= 2) {
            const id = parseInt(parts[0]);
            const title = parts[1];
            if (!isNaN(id) && title) {
                movies.push({ id, title });
            }
        }
    }
    
    return movies;
}

function parseRatingData(text) {
    const lines = text.split('\n').filter(line => line.trim());
    const ratings = [];
    
    for (const line of lines) {
        const parts = line.split('\t');
        if (parts.length >= 3) {
            const userId = parseInt(parts[0]);
            const movieId = parseInt(parts[1]);
            const rating = parseFloat(parts[2]);
            
            if (!isNaN(userId) && !isNaN(movieId) && !isNaN(rating)) {
                ratings.push({ userId, movieId, rating });
            }
        }
    }
    
    return ratings;
}
