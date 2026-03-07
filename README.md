# Enhanced Minority Game with Betting History Export

This is an enhanced version of the Minority Game implementation that tracks and exports each agent's betting history to a JSON file.

## Features

- **Betting History Tracking**: Each agent's bet for every round is recorded
- **JSON Export**: Complete betting history is exported to a JSON file after the game ends
- **Game Parameters**: All game configuration parameters are included in the JSON output
- **Agent Metadata**: Each agent's properties (naive, producer, ID) are recorded

## Compilation

\`\`\`bash
make
\`\`\`

## Usage

\`\`\`bash
./minority_game_enhanced [options]
\`\`\`

### Options

- `-n <num>`: Number of players (default: 101)
- `-m <num>`: Memory size (default: 3)
- `-s <num>`: Strategies per agent (default: 2)
- `-t <num>`: Equilibration time (default: 500)
- `-e <num>`: Number of naive players (default: 0)
- `-p <num>`: Number of producer players (default: 0)
- `--seed <num>`: Random seed (default: current time)
- `-r`: Run bidirectional game
- `-v`: Verbose mode
- `-h, --help`: Show help message

## Output

The program generates a JSON file (`betting_history.json` for regular games, `betting_history_bidirectional.json` for bidirectional games) containing:

- Game parameters (number of players, memory size, etc.)
- For each agent:
  - Agent ID
  - Whether the agent is naive
  - Whether the agent is a producer
  - Complete betting history (array of +1/-1 values)
  - Total number of bets made

## Example JSON Output

\`\`\`json
{
  "game_parameters": {
    "number_of_players": 101,
    "memory_size": 3,
    "number_of_strategies": 2,
    "naive_players": 0,
    "number_of_producers": 0,
    "equilibration_time": 500,
    "initial_seed": 1234567890,
    "alpha": 0.5
  },
  "agents": [
    {
      "agent_id": 0,
      "is_naive": false,
      "is_producer": false,
      "betting_history": [1, -1, 1, 1, -1, ...],
      "total_bets": 10500,
      "alpha": 0.5
    },
    ...
  ]
}
\`\`\`

## Changes Made

1. **Agent Class Enhancement**: Added `betting_history` vector to track all bets
2. **History Recording**: Modified `Bet()` method to record each bet
3. **JSON Export**: Added `ExportBettingHistoryToJSON()` method to the minority class
4. **Automatic Export**: Game automatically exports history after completion
5. **Enhanced Main**: Updated main program with better command-line interface

The betting history is recorded for all rounds including equilibration, giving you complete visibility into agent behavior throughout the entire game.
