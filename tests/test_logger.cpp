#include <iostream>
#include <sstream>
#include <cassert>
#include "logger.h"  
#include <fstream>

void getLastLoggedMessage(std::string& fileName, std::string& lastMessage){

    std::ifstream file(fileName, std::ios::ate);

    if(!file.is_open()){
        std::cout << "Error in opening file.";
    }

    std::streamoff pos = file.tellg();
    char ch;
    pos--;
    pos--;

    while(pos > 0){
        file.seekg(pos);
        file.get(ch);

        if(ch == '\n')
            break;

        pos--;
    }

    if(pos > 0)
        pos++;
        
    file.seekg(pos);
    std::getline(file, lastMessage);

}

void testErrorLogging(std::string& filename){

    Logger logger(filename);

    std::string message("Error Occured, I don't like this.");
    // Log a message with ERROR severity, which should be logged
    logger.log(ILogger::Severity::kERROR, message.c_str());
    // Check that the message was logged
    std::string lastLine;
    getLastLoggedMessage(filename, lastLine);

    std::cout << lastLine << std::endl;
    assert(lastLine == "[ERROR] Error Occured, I don't like this.");
    std::cout << "testErrorLogging passed." << std::endl;

}

void testErrorLogging2(){

    Logger logger;
    std::string message("Error Occured, I don't like this.");
    // Log a message with ERROR severity, which should be logged
    logger.log(ILogger::Severity::kERROR, message.c_str());
    // Check that the message was logged
    std::string lastLine, fileName = "main.log";
    getLastLoggedMessage(fileName, lastLine);

    std::cout << lastLine << std::endl;
    assert(lastLine == "[ERROR] Error Occured, I don't like this.");
    std::cout << "testErrorLogging passed." << std::endl;

}

int main(){

    std::string fileName("example.log");
    // testErrorLogging(fileName);
    testErrorLogging2();
}
