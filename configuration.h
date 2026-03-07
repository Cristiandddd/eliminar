/***************************************************************************
                          configuration.h  -  description
                             -------------------
    begin                : 16 April 2023
    last                 : 8 April 2024
    email                : eerams@yahoo.com
 ***************************************************************************/
 
/***************************************************************************
 *   Copyright (C) 2024 by Ernesto Estevez Rams   						   *
 *  eerams@yahoo.com 											   *
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
 
#ifndef _MINORITYCONFIG_H
#define _MINORITYCONFIG_H

#include <iostream>
#include <string>

#include "agent.h"
 
 /*! \file configuration.h
    \brief Global configurations set by #define statements
    */
 
#define HEADER "mgraph 0.5 2023, by EER.\n\
Send bug reports to eerams@yahoo.com\n\n"

#define USAGE "usage: mgraph [options] output\n\
Plays the minority game.\n\n\
-b|--startingm   value       ----> Initial memory. Defaults is a random value.\n\
-c|--strategies  value       ----> Number of strategies per player. Default is 2.\n\
-d|--seed        value       ----> Seed for the random generator. Default takes time as seed.\n\
-e|--naive       value       ----> Number of naive players. Default is 0.\n\
-h|--help                    ----> This message.\n\
-i|--initiala    value       ----> Initial number of nodes. Default is 3.\n\
-l|--alpha       value       ----> The alpha value.\n\
-n|--nodes       value       ----> Number of nodes. Mandatory.\n\
-o|--memory      value       ----> Memory size. Overrides the alpha value (-l).\n\
-p|--producers   value       ----> Number of producers players. Default is 0.\n\
-t|--teq         value       ----> Time to equilibrium in units of 2^M. Default is 500.\n\
-v|--verbose                 ----> Verbose mode.\n"

#define PRINTHEADER                    1
#define VERBOSE_DEFAULT                0
#define DEFAULT_NOPLAYERS             -1
#define DEFAULT_NOSTRATEGIES           2
#ifndef DEFAULT_MEMORY 
#define DEFAULT_MEMORY                -1
#endif
#define DEFAULT_NAIVE                  0
#define DEFAULT_PRODUCER               0
#define DEFAULT_TEQ                    500
#define DEFAULT_OVERLAPPING            1
#define DEFAULT_P                      0
#define DEFAULT_IMEM                   0
#define DEFAULT_SEED                  -1
#define DEFAULT_INITIALPLAYERS         3
#define DEFAULT_ALPHA                 -1           


struct minority_options { 
	std::string ofname;           // the output filename
    
	int header;                   // if 1(true) a header will be printed in front of each output value.
	unsigned long initial_mu;
    int number_of_strategies;
    long seed;
    int naive;
    int number_of_players;
    int producers;
    int teq;
    bool help;
    bool verbose;
    int initial_agents;
    int memory;
    double alpha;
    
	
	minority_options(void){
		header=                 PRINTHEADER;
        initial_mu=             DEFAULT_IMEM;
		verbose=                VERBOSE_DEFAULT;
        number_of_strategies=   DEFAULT_NOSTRATEGIES;
        seed=                   DEFAULT_SEED;
        naive=                  DEFAULT_NAIVE;
        number_of_players=      DEFAULT_NOPLAYERS;
        producers=              DEFAULT_PRODUCER;
        teq=                    DEFAULT_TEQ;
        initial_agents=         DEFAULT_INITIALPLAYERS;
        memory=                 DEFAULT_MEMORY;
        alpha=                  DEFAULT_ALPHA;
	}

    void PrintOptions(std::ostream & o){
        o << "Number of nodes: " << number_of_players << std::endl;
        o << "Initial nodes: " << initial_agents << std::endl;
        o << "Naive players: " << naive << std::endl;
        o << "Producer players: " << producers << std::endl;
        o << "memory size: " << memory << std::endl;
        o << "Initial memory: " << initial_mu << std::endl;
        o << "Alpha: " << alpha << std::endl;
        o << "Number of strategies: " << number_of_strategies << std::endl;
        o << "Time to equilibrium: " << teq << std::endl;
    }
};

#endif
