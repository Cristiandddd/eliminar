/***************************************************************************
                         rnd.cpp  -  description
                             -------------------
    begin                : 26 March 2016
    email                : estevez@imre.uh.cu
 ***************************************************************************/
 
/***************************************************************************
 *   Copyright (C) 2016 by Ernesto Estevez Rams   						   *
 *   estevez@imre.oc.uh.cu   											   *
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
#include "rnd.h"
 
static std::mt19937 rng;
static unsigned int rnd_number_of_call;
static std::mt19937 rng_saved_state;
 
long RNDInit(int seed){
	 time_t t=time(NULL);
	 long theseed=0;
	 
	 if(seed<0)
			theseed=t;
		else
			theseed=seed;
			
	rng.seed(theseed);
	rnd_number_of_call=0;
	
return theseed;
 }
 
 unsigned long RNDInteger(unsigned long max){
	 std::uniform_int_distribution<unsigned long> dist(0, max);
	 rnd_number_of_call++;
	 return dist(rng);
 }
 
 double RNDDouble(void){
	 std::uniform_real_distribution<double> dist(0.0, 1.0);
	 rnd_number_of_call++;
	 return dist(rng);
 }
 
 void RNDExit(void){
	 // Nothing to clean up with std::mt19937
 }
 
 unsigned int RNDNumberOfCalls(void){
 return rnd_number_of_call;
 }
 
 void RNDSaveState(void){
	rng_saved_state = rng;
 }
 
 void RNDRestoreState(void){
	rng = rng_saved_state;
 }
