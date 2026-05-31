#include "sinks/modes/FileDetectionSink.hpp"

// CONSTRUCTOR
FileDetectionSink::FileDetectionSink(bool saveNormalized) {
    m_saveNormalized = saveNormalized;
}

void FileDetectionSink::consumeSingle(PostProcessOutput& output, Logger& logger) {
  
    NVTX_RANGE("WRITE_DET");
    NVTX_RANGE("SERIALIZE_DETECTIONS");

    if (m_saveNormalized) {

        for ( auto& det : output.detections ) {
            if (!normalizeDetectionInPlace(det, output.metadata.outputWidth, output.metadata.outputHeight)) {
                logger.logConcatMessage(
                    Severity::kWARNING,
                    "Trying to normalize an already normalized detection, id= ",
                    det.metadata.detectionId,
                    " Image Path: ",
                    det.metadata.imgPath
                );
            }
        }
    }

    std::vector<uint8_t> bytes = serializeDetectionsToByteArray(output.detections);
    NVTX_POP();

    fs::path savePath = output.metadata.saveDetPath;

    if ( savePath.empty() ) {
        
        if ( output.metadata.imagePath.empty() || !std::filesystem::is_regular_file(output.metadata.imagePath) ) {
            throw std::runtime_error("Image Path not found: " + output.metadata.imagePath.string());
        }

        if ( output.metadata.resultsDir.empty() || !std::filesystem::is_directory(output.metadata.resultsDir) ) {
            throw std::runtime_error("Cannot Save results to: " + output.metadata.resultsDir.string() );
        }

        savePath = output.metadata.resultsDir / (output.metadata.imagePath.stem().string() + "_detection" + ".bin");

    }

    std::ofstream detFile(savePath, std::ios::out | std::ios::binary);

    if (!detFile) {
        throw std::runtime_error("Couldn't create the file");
    }
    detFile.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    detFile.close();
    NVTX_POP();
}