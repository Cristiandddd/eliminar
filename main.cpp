/***************************************************************************
           main_enhanced.cpp  -  Enhanced minority game with JSON export
                             -------------------
    begin                : 26 March 2024
    last                 : Modified for betting history export
    email                : eerams@yahoo.com
 ***************************************************************************/

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdlib>
#include <ctime>

#include "minority.h"
#include "configuration.h"
#include "rnd.h"

void printUsage(const char* program_name) {
    std::cout << "Enhanced Minority Game with Betting History Export\n";
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -n <num>     Number of players (default: 101)\n";
    std::cout << "  -m <num>     Memory size (default: 3)\n";
    std::cout << "  -s <num>     Strategies per agent (default: 2)\n";
    std::cout << "  -t <num>     Equilibration time (default: 500)\n";
    std::cout << "  -e <num>     Number of naive players (default: 0)\n";
    std::cout << "  -p <num>     Number of producer players (default: 0)\n";
    std::cout << "  --seed <num> Random seed (default: current time)\n";
    std::cout << "  -v           Verbose mode\n";
    std::cout << "  -h, --help   Show this help message\n\n";
    std::cout << "The game will automatically export betting history to 'betting_history.json'\n";
}

int main(int argc, char* argv[]) {
    minority_options opts;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-n" && i + 1 < argc) {
            opts.number_of_players = std::atoi(argv[++i]);
        } else if (arg == "-m" && i + 1 < argc) {
            opts.memory = std::atoi(argv[++i]);
        } else if (arg == "-s" && i + 1 < argc) {
            opts.number_of_strategies = std::atoi(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc) {
            opts.teq = std::atoi(argv[++i]);
        } else if (arg == "-e" && i + 1 < argc) {
            opts.naive = std::atoi(argv[++i]);
        } else if (arg == "-p" && i + 1 < argc) {
            opts.producers = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            opts.seed = std::atol(argv[++i]);
        } else if (arg == "-v") {
            opts.verbose = true;
        }
    }
    
    // Validate parameters
    if (opts.number_of_players <= 0) {
        std::cerr << "Error: Number of players must be positive\n";
        return 1;
    }
    if (opts.memory <= 0) {
        std::cerr << "Error: Memory size must be positive\n";
        return 1;
    }
    
    // Initialize random number generator
    RNDInit(opts.seed);
    
    if (opts.verbose) {
        std::cout << "Enhanced Minority Game Simulation\n";
        std::cout << "=================================\n";
        opts.PrintOptions(std::cout);
        std::cout << std::endl;
    }
    
    // Create and run the game
    minority game(opts);
    
    std::cout << "Running minority game simulation...\n";
    
    game.Run();
    
    std::cout << "Game completed. Betting history has been exported to JSON file.\n";
    
    return 0;
}
