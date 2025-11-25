# PART 1 - Parse pdf sheet music into mxl file using Audiveris

# Get arguments (user inputs: uploaded file path, tempo number)
PDF_FILE=$1

# Run Audiveris via Java
/Applications/Audiveris.app/Contents/MacOS/Audiveris -batch -export "$PDF_FILE"
# output final .mxl path to node
MXL_FILE="${PDF_FILE%.pdf}.mxl"
# Node can capture
echo "Generated MXL: $MXL_FILE"