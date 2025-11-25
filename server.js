// Import modules 
//(web framework, middleware for handling file uploads, work w/ file paths)
const express = require('express');
const multer = require('multer');
const path = require('path');
const { spawn } = require('child_process');

// creates Express app (web server, listen on port 3000)
const app = express();
const port = 3000;

// EJS setup
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// middleware to parse text fields from forms
app.use(express.urlencoded({ extended: true }));

// Configure Multer for disk storage
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads/'); // Files stored in the 'uploads/' folder
    },
    filename: function (req, file, cb) {
        // Generate new filename using timestamp and original extension
        cb(null, Date.now() + path.extname(file.originalname));
    }
});

// Create the Multer upload middleware
const upload = multer({ storage: storage }).fields([
    { name: 'pdfFile', maxCount: 1 },
    { name: 'audioFile', maxCount: 1 }
])

// Serve static files from the 'public' folder
app.use(express.static('public'));

// Route to handle file uploads
app.post('/upload', upload, (req, res) => {
    const tempo = req.body.tempo;

    if (!req.files || !req.files.pdfFile || !req.files.audioFile) { // if file not uploaded
       return res.status(400).send('PDF and audio files are required.');
    }

    // joins folder and filename into complete file path
    const pdfPath = path.join(__dirname, 'uploads', req.files.pdfFile[0].filename);
    const audioPath = path.join(__dirname, 'uploads', req.files.audioFile[0].filename);
    
    // set headers for live streaming
    res.setHeader('Content-Type', 'text/plain'); // tells browser is plain text
    res.setHeader('Transfer-Encoding', 'chunked'); // stream data in chunks

    // confirm what file/tempo user inputted
    res.write(`PDF uploaded: ${req.files.pdfFile[0].filename}\n`);
    res.write(`Audio uploaded: ${req.files.audioFile[0].filename}\n`);
    res.write(`Tempo: ${tempo}\n\n`);

    // Run Audiveris
    // path to shell script
    const audiverisScript = path.join(__dirname, 'scripts', 'polygence.sh');
    // run shell script with arguments ($1 = pdf file path)
    const audiverisProcess = spawn('bash', [audiverisScript, pdfPath])

    let mxlFilePath = '';

    // stream standard output of shell script to browser
    audiverisProcess.stdout.on('data', (data) => {
        const output = data.toString();
        res.write(output);


        // Capture last line containing .mxl file path
        // check if match[1] is absolute
        const match = output.match(/Generated MXL:\s*(.*)/);
        if (match) {
            // debug
            console.log("DEBUG: raw match ->", match[1]);

            let filePath = match[1].trim();

            // if audiveris gave a relative path, make it absolute
            if (!path.isAbsolute(filePath)) {
                filePath = path.join(__dirname, filePath);
            }
        
            mxlFilePath = filePath;
            res.write(`\nDetected MXL file: ${mxlFilePath}\n`);
        }
    });

    // stream error messages from script to browser
    audiverisProcess.stderr.on('data', (data) => {
        res.write(`Audiveris ERROR: ${data.toString()}`);
    });

    // close response when script finishes
    audiverisProcess.on('close', (code) => {
        res.write(`\nAudiveris exited with code ${code}\n`);

        if (!mxlFilePath) {
            res.write('No MXL file detected, stopping. \n');
            return res.end();
        }

        const fs = require('fs');
        if (!fs.existsSync(mxlFilePath)) {
            res.write(`ERROR: MXL file not found at ${mxlFilePath}\n`);
            return res.end();
        }

        res.write('\n--- Python CREPE Output ---\n');
        
        // Run python
        const pythonScript = path.join(__dirname, 'scripts', 'run_crepe.py')
        const pythonProcess = spawn('python3', [pythonScript, mxlFilePath, audioPath, tempo]);

        pythonProcess.stdout.on('data', (data) => res.write(data.toString()));
        pythonProcess.stderr.on('data', (data) => {console.error("Python STDERR:", data.toString())});
        
        pythonProcess.on('close', (pCode) => {
            res.write(`\nPython script exited with code ${pCode}\n`);
            res.end();
        })
    });
});

// Start the server
app.listen(port, () => {
    console.log(`Server listening on port ${port}`);
});