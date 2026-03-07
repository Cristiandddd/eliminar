/***************************************************************************
                         rnd.h  -  description
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
 #ifndef _RND_MINORITY_H
 #define _RND_MINORITY_H
 
 #include <ctime> 
 #include <cassert>
 #include <random>
 
long RNDInit(int seed=0);
unsigned long RNDInteger(unsigned long max);
double RNDDouble(void);
void RNDExit(void);
unsigned int RNDNumberOfCalls(void);
void RNDSaveState(void);
void RNDRestoreState(void);
 
 #endif
