/***************************************************************************
                          agent.h  -  description
                             -------------------
    begin                : 26 March 2023
	last                 : 4 April 2024
    email                : eerams@yahoo.com
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
 
 #ifndef _AGENT_H_
 #define _AGENT_H_
 
 #include <vector>
  #include "rnd.h"
  
  #define MAXITERATIONSBEFOREGIVINGUP  100000
 
 struct strategy {
	long score;
	std::vector<int> look_up_table;
 
	strategy(void){score=0;};
	strategy(const strategy & str){*this=str;};
	strategy(int p){look_up_table.assign(p,0);};
	
	strategy & operator=(const strategy & str){score=str.score; look_up_table=str.look_up_table; return *this;} 
	bool operator==(const strategy & str){return look_up_table==str.look_up_table;} 
 };
 
 class agent {
	protected:
		  bool producer;
		  bool naive;
		  bool frozen;
		  bool stationary;
          
          int id;
		  
		  int P;
		  int best_strategy;
		  std::vector<strategy> strategies; // lookup table for strategies
        
          int bet_record;
          std::vector<int> betting_history;

public:
    
	      agent(void);
	      agent(int ide, bool prod);;
	      agent(int ide, unsigned long p, int number_of_strategies, bool naiv=false, bool prod=false);
	      agent(const agent & ag){*this=ag;};
	      
	      int Initialize(int ide, unsigned long p, int number_of_strategies, bool naiv=false, bool prod=false);
	      void ClearRecords(void){  bet_record=0; 
                                    frozen=true;
                                    betting_history.clear();};
	      
	      int Bet(unsigned long mu, unsigned long mu_naive);
	      void UpdateScore(unsigned long mu, int A);
	      
	      bool Producer(void)const{return producer;};
	      bool Producer(bool p){producer=p; return producer;};
	      bool SetProducer(void){producer=true; return producer;};
	      bool Stationary(void)const{return stationary;};
	      bool SetStationary(void){stationary=true; return stationary;};
	      bool Frozen(void)const{return frozen;};
	      bool ResetFrozen(void){frozen=true; return frozen;};
	      bool Naive(void)const{return naive;};
		  bool DidIWin(int win);
	      int BestStrategy(void)const{return best_strategy;}
	      int SetBestStrategy(int bs){best_strategy=bs; return best_strategy;}
          
          int SetP(int p){P=p; return P;}
          int SetId(int ide){id=ide; return id;}
          int GetId() const { return id; }
          
          const std::vector<int>& GetBettingHistory(void) const {return betting_history;}
          void AddBetToHistory(int bet) {betting_history.push_back(bet);}
	      
	      agent & operator=(const agent & ag);
	      bool operator==(const agent & ag)const;
	      
	      
};

inline agent::agent(void){
	producer=false; 
	frozen=true; 
	stationary=false; 
	best_strategy=0;
	naive=false;
};

inline agent::agent(int ide, bool prod){
	producer=prod; 
	frozen=true; 
	stationary=false;
	best_strategy=0;
	naive=false;
    id=ide;
	};
	
inline agent::agent(int ide, unsigned long p, int number_of_strategies, bool naiv, bool prod){
	Initialize(ide, p, number_of_strategies, naiv, prod);
}

inline bool agent::operator==(const agent & ag)const{
	bool flag=true;
	
	flag=(producer==ag.producer) && flag;
	flag=(naive==ag.naive) && flag;
	flag=(frozen==ag.frozen) && flag;
	flag=(stationary==ag.stationary) && flag;
		  
	flag=(P==ag.P) && flag;
	flag=(best_strategy == ag.best_strategy) && flag;
	
	std::vector<strategy>::const_iterator it=ag.strategies.begin();
	for(auto str : strategies){
	   if (((flag=(str==*it)) && flag)==false)
	      break;
	   it++;
	}
	
return flag;	
}


 
 
 #endif
