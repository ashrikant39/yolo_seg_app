#include <iostream>
#include <sstream>
#include <cassert>
#include "logger.h"  
#include <fstream>

void testErrorLogging() 
{
    std::stringstream ss;
    Logger logger(ss);

    // Log a message with ERROR severity, which should be logged
    logger.log(NVLogger::Severity::kERROR, "Error Occured, I don't like this.");

    // Check that the message was logged
    assert(ss.str() == "[ERROR] Error Occured, I don't like this.\n");
    std::cout << "testErrorLogging passed." << std::endl;
}

void testInfoLoggingIgnored() 
{
    std::stringstream ss;
    Logger logger(ss, NVLogger::Severity::kWARNING);

    // Log a message with INFO severity, which should be ignored
    logger.log(NVLogger::Severity::kINFO, "Info. I need this info.");

    // Check that nothing was logged
    assert(ss.str().empty());
    std::cout << "testInfoLoggingIgnored passed." << std::endl;
}

void testWarningLogging() 
{
    std::stringstream ss;
    Logger logger(ss);

    // Log a message with WARNING severity, which should be logged
    logger.log(NVLogger::Severity::kWARNING, "This is a warning.");

    // Check that the message was logged
    assert(ss.str() == "[WARNING] This is a warning.\n");
    std::cout << "testWarningLogging passed." << std::endl;
}

void testCustomStream() 
{
    std::stringstream ss;
    Logger logger(ss);
    // Log an error to the custom stream
    logger.log(NVLogger::Severity::kERROR, "Custom stream message");

    // Check that the custom stream received the log
    assert(ss.str() == "[ERROR] Custom stream message\n");
    std::cout << "testCustomStream passed." << std::endl;
}

void testLoggingToFile(const std::string& logFileName) 
{
    Logger logger(logFileName);
    // Log an error to the custom stream
    logger.log(NVLogger::Severity::kERROR, "Custom stream message1");
    logger.log(NVLogger::Severity::kWARNING, "Custom stream message2");
    logger.log(NVLogger::Severity::kINFO, "Custom stream message3");

    std::ifstream logFileStream(logFileName);

    if(!logFileStream.is_open())
    {
        std::cerr << "Failed to open the log file.\n";
        std::cout << "testCustomStream failed." << std::endl;
        return;
    }

    std::string line;
    
    std::getline(logFileStream, line);
    assert(line == "[ERROR] Custom stream message1");

    std::getline(logFileStream, line);
    assert(line == "[WARNING] Custom stream message2");

    std::getline(logFileStream, line);
    assert(line == "[INFO] Custom stream message3");

    // Check that the custom stream received the log
    
    std::cout << "testCustomStream passed." << std::endl;
}

void testConsoleLogging() 
{
    // Create a stringstream to capture the output
    std::stringstream capturedOutput;
    
    // Save the original buffer of std::cout
    std::streambuf* originalCoutBuffer = std::cout.rdbuf();

    // Redirect std::cout to the stringstream
    std::cout.rdbuf(capturedOutput.rdbuf());

    // Perform the logging operation that outputs to std::cout
    Logger logger;  // Assuming Logger takes std::ostream for output
    logger.log(NVLogger::Severity::kWARNING, "This is a warning.");

    // Restore std::cout to its original buffer
    std::cout.rdbuf(originalCoutBuffer);

    // Verify the captured output
    assert(capturedOutput.str() == "[WARNING] This is a warning.\n");
    std::cout << "testConsoleLogging passed." << std::endl;
}

int main(int argc, char** argv) 
{
    testErrorLogging();
    testInfoLoggingIgnored();
    testWarningLogging();
    testCustomStream();
    testConsoleLogging();
    if(argc > 1)
        testLoggingToFile(argv[1]);

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
