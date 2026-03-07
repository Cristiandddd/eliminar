/***************************************************************************
                          minority.h  -  description
                             -------------------
    begin                : 26 March 2023
	last                 : 8 April 2024
    email                : erams@yahoo.com
 ***************************************************************************/
 
/***************************************************************************
 *   Copyright (C) 2024 by Ernesto Estevez Rams   						   *
 *   eerams@yahoo.com  											   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
 
#ifndef _MINORITY_H_
#define _MINORITY_H_

#include <vector>
#include <string>
 
#include "agent.h"
#include "rnd.h"
#include "configuration.h"
 

#define DEFAULT_NOPLAYERS             -1
#define DEFAULT_NOSTRATEGIES           2
#define DEFAULT_MEMORY                 -1
#define DEFAULT_NAIVE                  0
#define DEFAULT_PRODUCER               0
#define DEFAULT_NOROUNDS               300
#define DEFAULT_TEQ                    500
#define DEFAULT_OVERLAPPING            1
#define DEFAULT_P                      0
#define DEFAULT_IMEM                   0
#define DEFAULT_SEED                  -1
#define DEFAULT_INITIALPLAYERS         3
#define DEFAULT_INITIAL_P              10
// #define DEFAULT_ALPHA                  0.5

 
 struct minority_parameters { 
	int number_of_players; 
    int initial_players;
	int number_of_strategies; 
	int teq; 
	int memory; 
	//int rounds; 
	int naive; 
	int producers; 
	unsigned long P; 
	unsigned long initial_mu; 
	long seed; 
	double alpha;
	
	minority_parameters(void){
		number_of_players=DEFAULT_NOPLAYERS;
        initial_players=DEFAULT_INITIALPLAYERS;
		number_of_strategies=DEFAULT_NOSTRATEGIES;
		teq=DEFAULT_TEQ;
		memory=DEFAULT_MEMORY;
		naive=DEFAULT_NAIVE;
		producers=DEFAULT_PRODUCER;
		P=DEFAULT_P;
		initial_mu=DEFAULT_IMEM;
		seed=DEFAULT_SEED;
		alpha=DEFAULT_ALPHA;
	}
};
 
 
 class minority {
	 protected:
		int number_of_players;
        int initial_agents;
		int naive_players;
		int number_of_producers;
		int number_of_strategies;
		int teq;
		int memory;
		unsigned long initial_mu; // initial memory
		long initial_seed;
		double alpha;
		
		std::vector<agent> players;
	 
	public:
		minority(void);
		minority(struct minority_parameters & mino);
        minority(struct minority_options & mino);
		minority(const minority & mi);
		
		void Initialize(void);
		void Clear(void);
		int Run(void);

		
		void ExportBettingHistoryToJSON(const std::string& filename) const;

		
		int NumberOfPlayers(void)const{return number_of_players;};
		int NumberOfProducers(void)const{return number_of_producers;};
		int NumberOfNaivePlayers(void)const{return naive_players;};
		int NumberOfStrategies(void)const{return number_of_strategies;};
		int StationaryTime(void)const{return teq;};
		int Memory(void)const{return memory;};
		unsigned long InitialMemory(void)const{return initial_mu;};
		long Seed(void)const{return initial_seed;};
		double Alpha(void)const{return alpha;};
	
		
		double ProducersFraction(void)const{return (double)number_of_producers/(double)number_of_players;};
		double NaiveFraction(void)const{return (double)naive_players/(double)number_of_players;};
		
		minority & operator=(const minority & mi);
		minority & operator=(struct minority_parameters & mino);
		bool operator==(const minority & mi)const;
        
        std::vector<agent> GetPlayers(void)const{return players;};
        unsigned int PlayersSize(void)const{return players.size();};
        agent & Player(unsigned int index){return players[index];};
};


inline minority::minority(void){
initial_agents=DEFAULT_INITIALPLAYERS;
memory=DEFAULT_MEMORY;
number_of_players=naive_players=number_of_producers=number_of_strategies=teq=memory=0;
alpha=DEFAULT_ALPHA;
}


inline bool minority::operator==(const minority & mi)const{
bool flag=true;

		flag=(number_of_players==mi.number_of_players);
		flag=(naive_players==mi.naive_players) && flag;
		flag=(number_of_producers==mi.number_of_producers) && flag;
		flag=(number_of_strategies==mi.number_of_strategies) && flag;
		flag=(teq==mi.teq) && flag;
		flag=(memory==mi.memory) && flag;
		flag=(initial_mu==mi.initial_mu) && flag; // initial memory
		flag=(initial_seed==mi.initial_seed) && flag;
		flag=(players==mi.players) && flag;

return flag;
}

#endif
