/***************************************************************************
                          binsequence.h  -  description
                             -------------------
    begin                : 16 Nov 2013
    email                : estevez@imre.oc.uh.cu
 ***************************************************************************/
 
/***************************************************************************
 *   Copyright (C) 2013 by Ernesto Estevez Rams   						   *
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
 
 #ifndef _BINARY_SEQUENCE_H
 #define _BINARY_SEQUENCE_H
 
 #include <iostream>
 #include <algorithm>
 #include <vector>
 #include <set>
 #include <climits>
 #include <string>
 #include <functional>
 
 #include "myexceptions.h"
 
 #ifdef ALPHABETSIZE
 #undef ALPHABETSIZE 
 #endif
 
 #define ALPHABETSIZE 0x02
 
  // error flags
 enum binseq_error {binseq_ok=0, binseq_nomemory, binseq_unknown, binseq_outofbounds, binseq_sizenomatch};
 
// The error condition exceptions
 class BinarySequenceError : Errors {};
 class BinarySequenceBadAlloc: public BinarySequenceError , public BadAlloc {};
 class BinarySequenceNoMatchSize: public BinarySequenceError , public NotEqualSize {};
 class BinarySequenceOutOfBounds: public BinarySequenceError , public OutOfBounds {};
 class BinarySequenceUnknownError: public BinarySequenceError , public UnknownError {};
//...............................................

class sequence;

//.....................................................
//              The binary sequence class
//.....................................................
/// \brief The binary sequence class
class binsequence {
	
	protected:
		std::vector<bool> seq;                                                                //!< the sequence.
		binseq_error error;                                                                   //!< error flag.
		
	public:	
		binsequence(void):error(binseq_ok){};                                                 //!< default constructor.
		binsequence(const binsequence & s);  
		binsequence(const sequence & s);
		binsequence(const std::vector<bool> & s);                                             //!< copy constructor.
		binsequence(const std::vector<char> & s);                                                   //!< copy constructor.
		binsequence(const std::vector<int> & s); 													  //!< copy constructor.
		binsequence(const std::string & s);                                                         //!< copy constructor.
		~binsequence();								                                          //!< destructor.
		
		int Error(void)const{return error;};                                                  //!< the error consdition of the object.
		int clearerrorflag(void){return error=binseq_ok;};                                    //!< reset the error flag.
		
		
		std::vector<bool>::reference operator[](std::vector<bool>::size_type );               //!< subscript operator. Bounds checked.
		std::vector<bool>::reference first(void);                                             //!< Reference to first element.
		std::vector<bool>::reference last(void);                                              //!< Reference to last element.
		
		std::vector<bool>::reference at(std::vector<bool>::size_type);     					  //!< access operator. Bounds not checked.
		bool const_at(std::vector<bool>::size_type index)const;                  				  //!< access operator. Bounds not checked.
		
		std::vector<bool>::size_type  push(bool);                                             //!< push stack operation.
		std::vector<bool>::size_type  push(char);                                             //!< push stack operation.
		std::vector<bool>::size_type  push(int);                                              //!< push stack operation.
		std::vector<bool>::size_type  pop(void);                                              //!< pop stack operation.
		std::vector<bool>::reference  back(void);                                              //!< returns the last value in the stack. No reference value.
		
		std::vector<bool>::size_type size(void)const;                                         //!< length of the sequence.
		std::vector<bool>::size_type length(void)const{return size();};                       //!< length of the sequence.
		
		std::vector<bool>::iterator  begin(void){return seq.begin();};
		std::vector<bool>::iterator  end(void){return seq.end();};
		
		std::vector<bool> SequenceVector(void);                                               //!< returns the actual sequence vector.
		
		binsequence  & pi(void);                                                              //!< the pi (lempel-ziv) operator. Drops the last element of the sequence.
		binsequence  & most(void){return pi();};                                              //!< Drops the last element of the sequence.
		binsequence  & reverse(void);                                                         //!< Reverse the sequence.
		binsequence  & negate(void);
		binsequence  & rightshift(std::vector<bool>::size_type ls=1);                         //!< Makes a right circular shift.
		binsequence  & leftshift(std::vector<bool>::size_type ls=1);                          //!< Makes a left circular shift.
		
		binsequence Subsequence(std::vector<bool>::size_type , unsigned int)const;
		
		double cyclicity(void);															  //!< Computes the hexagonality assuming Hagg code
		std::vector<int> rle(void);                                                       //!< Performs RLE encoding
		unsigned long NoZero(void)const;
		unsigned long NoOnes(void)const;
		unsigned long Search(const binsequence &, unsigned long)const;
		unsigned long PatternFrequency(const binsequence & pat) const;
		unsigned long ExclusivePatternFrequency(const binsequence & pat) const;
		unsigned long ExclusiveBraceletFrequency(const binsequence & pat) const;
		
		
		binsequence & operator=(binsequence);                                                 //!< assign operator.
		binsequence & operator=(const std::vector<bool> &);                                   //!< assign operator.
		binsequence & operator=(const std::vector<char> &);                                   //!< assign operator.
		binsequence & operator=(const std::vector<int> &);                                    //!< assign operator.
		binsequence & operator=(const std::string &);                                         //!< assign operator.
		binsequence & operator|=(const binsequence &);                                        //!< bitwise or operator.
		binsequence & operator&=(const binsequence &);                                        //!< bitwise and operator.
		binsequence & operator^=(const binsequence &);                                        //!< bitwise xor operator (Hamming field).
		binsequence & operator+=(const binsequence &);                                        //!< bitwise or operator.
		binsequence & operator-=(const binsequence &);                                        //!< bitwise xor operator (Hamming field).
		
		binsequence & ToBlackAndWhite(const sequence &);
		
		void clear();                                                                         //!< clears the object.
		
		bool bit(std::vector<bool>::size_type)const;                                          //!< Test the value of the bit given by its index
		bool flip(std::vector<bool>::size_type);                                              //!< Flips value in the sequence
		void set(std::vector<bool>::size_type);                                               //!< Set to true a value in the sequence
		void unset(std::vector<bool>::size_type);                                             //!< Sets to false a value in the sequence
		
		// friend functions
		
		friend std::ostream & operator<<(std::ostream & , const binsequence &);               //!< saving the banana.
		
		friend void swap(binsequence &, binsequence & );                                      //!< swap two objects.
		
		friend bool operator==(const binsequence &, const binsequence &);                     //!< comparison operator.
		friend bool operator!=(const binsequence &, const binsequence &);                     //!< comparison operator.
		friend bool operator>(const binsequence &, const binsequence &);                      //!< comparison operator.
		friend bool operator>=(const binsequence &, const binsequence &);                     //!< comparison operator.
		friend bool operator<(const binsequence &, const binsequence &);                      //!< comparison operator.
		friend bool operator<=(const binsequence &, const binsequence &);                     //!< comparison operator.
		
		friend binsequence operator|(const binsequence &, binsequence &);                     //!< bitwise or operator.
		friend binsequence operator&(const binsequence &, binsequence &);                     //!< bitwise and operator.
		friend binsequence operator^(const binsequence &, binsequence &);                     //!< bitwise xor operator (Hamming field).
		friend binsequence operator+(const binsequence &, binsequence &);                     //!< bitwise or operator.
		friend binsequence operator-(const binsequence &, binsequence &);                     //!< bitwise xor operator (Hamming field).
		friend binsequence operator!(const binsequence &);                                    //!< bitwise not operator.
			
		friend binsequence map(std::function<bool(bool)> fn, const binsequence & s);          //!< applies a function (functor) to every member of the sequence.
		friend binsequence for_each(std::function<bool(bool)> fn, const binsequence & s);     //!< applies a function (functor) to every member of the sequence.
		
		friend binsequence OnePointSequence(binsequence &, std::vector<bool>::size_type);			  //!< Build a sequence of a given length with a '1' at the middle and '0' on the rest
		friend unsigned long HammingDistance(binsequence &, binsequence &);       //!< Computes the Hamming distance
		
		friend binsequence Reverse(const binsequence &);
		friend binsequence Negate(const binsequence &);
		friend std::set<binsequence>  BuildBraceletSet(const binsequence &);
};

binsequence map(std::function<bool(bool)>  fn, const binsequence & s);
binsequence OnePointSequence(binsequence & bs, std::vector<bool>::size_type length);
binsequence Negate(const binsequence & bs);
binsequence Reverse(const binsequence & bs);
unsigned long HammingDistance(binsequence & op, binsequence & op1);
std::set<binsequence>  BuildBraceletSet(const binsequence &bs);
std::vector<int> BoolToInt(const std::vector<bool> &);

//.........................  Member functions ................................


// .........................  Constructors ....................................
// .............................................................................
// Name: constructor functions
//
// Synopsis: copy constructor 
//
// Parameters: 
//			 const binsequence & s        -------> The source 
//		
// Returns:  Nothing
//   
// Exceptions: 
//            BinarySequenceBadAlloc      --------> Memory allocation failed. error flag is also set.
//            BinarySequenceError         --------> Generic error. error flag is also set.
//.............................................................................. 
inline binsequence::binsequence(const binsequence & s):error(s.error){
		error=binseq_ok;
		try{
			seq=s.seq;
			}
		catch (std::bad_alloc & ba){
			error=binseq_nomemory;
			throw BinarySequenceBadAlloc();
			}
		catch(...){
			error=binseq_unknown;
			throw BinarySequenceError();
		}
	}
	

// .............................................................................
// Name: constructor functions
//
// Synopsis: Constructor 
//
// Parameters: 
//			 const std::vector<bool> & s       -------> The source is a vector of bool values
//		
// Returns:  Nothing
//   
// Exceptions: 
//            BinarySequenceBadAlloc      --------> Memory allocation failed. error flag is also set.
//            BinarySequenceError         --------> Generic error. error flag is also set.
//.............................................................................. 	
inline binsequence::binsequence(const std::vector<bool> & s){
		error=binseq_ok;
		try{
			seq=s;
			}
		catch (std::bad_alloc & ba){
			error=binseq_nomemory;
			throw BinarySequenceBadAlloc();
			}
		catch(...){
			error=binseq_unknown;
			throw BinarySequenceError();
		}
	}

// .............................................................................
// Name: constructor functions
//
// Synopsis: Constructor 
//
// Parameters: 
//			 const std::vector<char> & s       -------> The source is a vector of char values
//
// Notes:
//      i) Even values are mapped to 0; Odd values are mapped to 1
//		
// Returns:  Nothing
//   
// Exceptions: 
//            BinarySequenceBadAlloc      --------> Memory allocation failed. error flag is also set.
//            BinarySequenceError         --------> Generic error. error flag is also set.
//..............................................................................	
inline binsequence::binsequence(const std::vector<char> & s){
		error=binseq_ok;
		try{
			seq.reserve(s.size());
			for (auto c : s)
			    seq.push_back((c%2) != 0);
			}
		catch (std::bad_alloc & ba){
			error=binseq_nomemory;
			throw BinarySequenceBadAlloc();
			}
		catch(...){
			error=binseq_unknown;
			throw BinarySequenceError();
		}
	}
	
// .............................................................................
// Name: constructor functions
//
// Synopsis: Constructor 
//
// Parameters: 
//			 const std::vector<char> & s       -------> The source is a string
//
// Notes:
//      i) Even values are mapped to 0; Odd values are mapped to 1
//		
// Returns:  Nothing
//   
// Exceptions: 
//            BinarySequenceBadAlloc      --------> Memory allocation failed. error flag is also set.
//            BinarySequenceError         --------> Generic error. error flag is also set.
//..............................................................................	
inline binsequence::binsequence(const std::string & s){
		error=binseq_ok;
		try{
			seq.reserve(s.size());
			for (std::string::size_type i=0; i < s.size(); ++i)
			    seq.push_back( ((s[i]-'0')%2) !=0 );
			}
		catch (std::bad_alloc & ba){
			error=binseq_nomemory;
			throw BinarySequenceBadAlloc();
			}
		catch(...){
			error=binseq_unknown;
			throw BinarySequenceError();
		}
	}	

// .............................................................................
// Name: constructor functions
//
// Synopsis: Constructor 
//
// Parameters: 
//			 const std::vector<int> & s       -------> The source is a vector of int values
//
// Notes:
//      i) Even values are mapped to 0; Odd values are mapped to 1
//		
// Returns:  Nothing
//   
// Exceptions: 
//            BinarySequenceBadAlloc      --------> Memory allocation failed. error flag is also set.
//            BinarySequenceError         --------> Generic error. error flag is also set.
//..............................................................................
inline binsequence::binsequence(const std::vector<int> & s){
		error=binseq_ok;
		try{
			seq.reserve(s.size());
			for (auto c : s)
			    seq.push_back((c%2) != 0);
			}
		catch (std::bad_alloc & ba){
			error=binseq_nomemory;
			throw BinarySequenceBadAlloc();
			}
		catch(...){
			error=binseq_unknown;
			throw BinarySequenceError();
		}
	}

// .............................................................................
// Name: destructor functions
//
// Synopsis: Destructor
//
// Parameters: 
//           None
//
// Returns:  Nothing
//   
// Exceptions: None
//..............................................................................
inline binsequence::~binsequence(){
	error=binseq_ok; seq.clear();
};


//............................................................................
//                Operator overloading
//.............................................................................

// .............................................................................
// Name: operator=
//
// Synopsis: Copy operator 
//
// Parameters: 
//			 binsequence s       -------> The source.
//		
// Returns: 
//         *this
//   
// Exceptions: 
//            BinarySequenceBadAlloc      --------> Memory allocation failed. error flag is not set.
//            BinarySequenceError         --------> Generic error. error flag is not set.
//..............................................................................
inline binsequence & binsequence::operator=(binsequence s){
    swap(*this, s);

    return *this;
}	

// .............................................................................
// Name: operator=
//
// Synopsis: Copy operator 
//
// Parameters: 
//			 const std::vector<bool> & s      -------> The source is  a vector of bool.
//		
// Returns: 
//         *this
//   
// Exceptions: 
//            BinarySequenceBadAlloc      --------> Memory allocation failed. error flag is set.
//            BinarySequenceError         --------> Generic error. error flag is set.
//..............................................................................
inline binsequence & binsequence::operator=(const std::vector<bool> & s){

		error=binseq_ok;
		try{
			seq=s;
			}
		catch (std::bad_alloc & ba){
			error=binseq_nomemory;
			throw BinarySequenceBadAlloc();
			}
		catch(...){
			error=binseq_unknown;
			throw BinarySequenceError();
		}

    return *this;
}	

// .............................................................................
// Name: operator=
//
// Synopsis: Copy operator 
//
// Parameters: 
//			 const std::vector<int> & s      -------> The source is  a vector of int.
// Notes:
//      i) Even values are mapped to 0; Odd values are mapped to 1
//		
// Returns: 
//         *this
//   
// Exceptions: 
//            BinarySequenceBadAlloc      --------> Memory allocation failed. error flag is set.
//            BinarySequenceError         --------> Generic error. error flag is set.
//..............................................................................
inline binsequence & binsequence::operator=(const std::vector<int> & s){

		error=binseq_ok;
		try{
			seq.clear();
			seq.reserve(s.size());
			for (int c : s)
			    seq.push_back(c%2);
			}
		catch (std::bad_alloc & ba){
			error=binseq_nomemory;
			throw BinarySequenceBadAlloc();
			}
		catch(...){
			error=binseq_unknown;
			throw BinarySequenceError();
		}

    return *this;
}	

// .............................................................................
// Name: operator=
//
// Synopsis: Copy operator 
//
// Parameters: 
//			 const std::vector<char> & s      -------> The source is  a vector of char.
// Notes:
//      i) Even values are mapped to 0; Odd values are mapped to 1
//		
// Returns: 
//         *this
//   
// Exceptions: 
//            BinarySequenceBadAlloc      --------> Memory allocation failed. error flag is set.
//            BinarySequenceError         --------> Generic error. error flag is set.
//..............................................................................
inline binsequence & binsequence::operator=(const std::vector<char> & s){

		error=binseq_ok;
		try{
			seq.clear();
			seq.reserve(s.size());
			for (auto c : s)
			    seq.push_back(c%2);
			}
		catch (std::bad_alloc & ba){
			error=binseq_nomemory;
			throw BinarySequenceBadAlloc();
			}
		catch(...){
			error=binseq_unknown;
			throw BinarySequenceError();
		}

    return *this;
}

// .............................................................................
// Name: operator=
//
// Synopsis: Copy operator 
//
// Parameters: 
//			 const std::string & s      -------> The source is  a string.
// Notes:
//      i) Even values are mapped to 0; Odd values are mapped to 1
//		
// Returns: 
//         *this
//   
// Exceptions: 
//            BinarySequenceBadAlloc      --------> Memory allocation failed. error flag is set.
//            BinarySequenceError         --------> Generic error. error flag is set.
//..............................................................................
inline binsequence & binsequence::operator=(const std::string & s){

		error=binseq_ok;
		try{
			seq.clear();
			seq.reserve(s.size());
			for (std::string::size_type i=0; i<s.size(); ++i)
			    seq.push_back( (s[i]-'0')%2);
			}
		catch (std::bad_alloc & ba){
			error=binseq_nomemory;
			throw BinarySequenceBadAlloc();
			}
		catch(...){
			error=binseq_unknown;
			throw BinarySequenceError();
		}

    return *this;
}

// .............................................................................
// Name: operator&=
//
// Synopsis: bitwise "and" operator 
//
// Parameters: 
//			 const binsequence & op      -------> comparing object
//		
// Returns: 
//         *this
//   
// Exceptions: 
//            BinarySequenceNoMatchSize   --------> Sizes of both operands do not match. error flag is set.
//            BinarySequenceError         --------> Generic error. error flag is set.
//..............................................................................
inline binsequence & binsequence::operator&=(const binsequence & op){

	if(seq.size() != op.size()){
	    error=binseq_sizenomatch;
	    throw BinarySequenceNoMatchSize();
		} else {
		try{
			 std::vector<bool>::const_iterator it1=op.seq.begin();
			
			 for (std::vector<bool>::iterator it = seq.begin(); it!=seq.end(); ++it){
			     *it=(*it) && (*it1);
			     ++it1;
				 }
			}
		catch(...){
			error=binseq_unknown;
			throw BinarySequenceError();
		}
	}
	
return *this;	
}

// .............................................................................
// Name: operator|=
//
// Synopsis: bitwise "or" operator 
//
// Parameters: 
//			 const binsequence & op      -------> comparing object
//		
// Returns: 
//         *this
//   
// Exceptions: 
//            BinarySequenceNoMatchSize   --------> Sizes of both operands do not match. error flag is set.
//            BinarySequenceError         --------> Generic error. error flag is set.
//..............................................................................
inline binsequence & binsequence::operator|=(const binsequence & op){

	if(seq.size() != op.size()){
	    error=binseq_sizenomatch;
	    throw BinarySequenceNoMatchSize();
		} else {
		try{
			 std::vector<bool>::const_iterator it1=op.seq.begin();
			
			 for (std::vector<bool>::iterator it = seq.begin(); it!=seq.end(); ++it){
			     *it=(*it) || (*it1);
			     ++it1;
				 }
			}
		catch(...){
			error=binseq_unknown;
			throw BinarySequenceError();
		}
	}
	
return *this;	
}

// .............................................................................
// Name: operator^=
//
// Synopsis: bitwise "xor" operator. Hamming field. 
//
// Parameters: 
//			 const binsequence & op      -------> comparing object
//		
// Returns: 
//         *this
//   
// Exceptions: 
//            BinarySequenceNoMatchSize   --------> Sizes of both operands do not match. error flag is set.
//            BinarySequenceError         --------> Generic error. error flag is set.
//..............................................................................
inline binsequence & binsequence::operator^=(const binsequence & op){

	if(seq.size() != op.size()){
	    error=binseq_sizenomatch;
	    throw BinarySequenceNoMatchSize();
		} else {
		try{
			 std::vector<bool>::const_iterator it1=op.seq.begin();
			
			 for (std::vector<bool>::iterator it = seq.begin(); it!=seq.end(); ++it){
			     *it=((*it)==(*it1))? false : true;
			     ++it1;
				 }
			}
		catch(...){
			error=binseq_unknown;
			throw BinarySequenceError();
		}
	}
	
return *this;	
}

// .............................................................................
// Name: operator+=
//
// Synopsis: bitwise "or" operator 
//
// Parameters: 
//			 const binsequence & op      -------> comparing object
//		
// Returns: 
//         *this
//   
// Exceptions: 
//            BinarySequenceNoMatchSize   --------> Sizes of both operands do not match. error flag is set.
//            BinarySequenceError         --------> Generic error. error flag is set.
//..............................................................................
inline binsequence & binsequence::operator+=(const binsequence & op){

return operator|=(op);
}

// .............................................................................
// Name: operator|=
//
// Synopsis: bitwise "xor" operator. Hamming field. 
//
// Parameters: 
//			 const binsequence & op      -------> comparing object
//		
// Returns: 
//         *this
//   
// Exceptions: 
//            BinarySequenceNoMatchSize   --------> Sizes of both operands do not match. error flag is set.
//            BinarySequenceError         --------> Generic error. error flag is set.
//..............................................................................
inline binsequence & binsequence::operator-=(const binsequence & op){

return operator^=(op);
}

// .............................................................................
// Name: operator[]
//
// Synopsis: indexing operator
//
// Parameters: 
//			 std::vector<bool>::size_type index      -------> the index of the requested value
//		
// Returns: 
//         std::vector<bool>::reference ------> the reference to the index value of the underlying bool vector
//   
// Exceptions: 
//            BinaryOutOfBounds           --------> Index is out of bounds. error flag is set.
//            BinarySequenceError         --------> Generic error. error flag is set.
//..............................................................................
inline std::vector<bool>::reference  binsequence::operator[](std::vector<bool>::size_type index){
	
	std::vector<bool>::size_type i=index;
	error=binseq_ok;
	
	if(index >= seq.size()){
	    i=seq.size()-1;
	    error=binseq_outofbounds;
	    throw BinarySequenceOutOfBounds();
		}

return seq[i];	 
}

// .............................................................................

// .............................................................................
// Name: SequenceVector
//
// Synopsis: Returns the underlying vector of bools 
//
// Parameters: 
//			 None
//		
// Returns: 
//         std::vector<bool>   ------>  The vector of bools. No reference return.
//   
// Exceptions: 
//            None
//..............................................................................
inline std::vector<bool> binsequence::SequenceVector(void){
return seq;
}

// .............................................................................
// Name: at
//
// Synopsis: indexing operator. No bounds check
//
// Parameters: 
//			 std::vector<bool>::size_type index      -------> the index of the requested value
//		
// Returns: 
//         std::vector<bool>::reference ------> the reference to the index value of the sequence
//   
// Exceptions: 
//            None
//..............................................................................
inline std::vector<bool>::reference  binsequence::at(std::vector<bool>::size_type index){		 
return seq.at(index);	
}

// .............................................................................
// Name: const_at
//
// Synopsis: indexing operator. No bounds check
//
// Parameters: 
//			 std::vector<bool>::size_type index      -------> the index of the requested value
//		
// Returns: 
//         bool ------> the index value of the sequence
//   
// Exceptions: 
//            None
//..............................................................................
inline bool binsequence::const_at(std::vector<bool>::size_type index)const{	
	return seq.at(index);
}

// .............................................................................
// Name: first
//
// Synopsis: returns a reference to the first value of the sequence
//
// Parameters: 
//			 None
//		
// Returns: 
//         std::vector<bool>::reference ------> the reference to first sequence value
//   
// Exceptions: 
//            None
//..............................................................................
inline std::vector<bool>::reference  binsequence::first(void){
	return seq.front();
}

// .............................................................................
// Name: last
//
// Synopsis: returns a reference to the last value of the sequence
//
// Parameters: 
//			 None
//		
// Returns: 
//         std::vector<bool>::reference ------> the reference to last sequence value
//   
// Exceptions: 
//            None
//..............................................................................
inline std::vector<bool>::reference binsequence::last(void){
	return seq.back();
}

// .............................................................................
// Name: back
//
// Synopsis: returns a reference to the last value of the sequence
//
// Parameters: 
//			 None
//		
// Returns: 
//         std::vector<bool>::reference ------> the reference to last sequence value
//   
// Exceptions: 
//            None
//..............................................................................
inline std::vector<bool>::reference  binsequence::back(void){
	return last();
}

// .............................................................................
// Name: push
//
// Synopsis: Adds a value to the end of the sequence
//
// Parameters: 
//			 bool c    -----> The added value
//		
// Returns: 
//         std::vector<bool>::size_type ------> the size of the sequence
//   
// Exceptions: 
//            BinarySequenceBadAlloc      --------> Memory allocation failed. error flag is set.
//            BinarySequenceError         --------> Generic error. error flag is set.
//..............................................................................
inline std::vector<bool>::size_type binsequence::push(bool c){
	try{
		seq.push_back(c);
		}
	catch (std::bad_alloc & ba){
			error=binseq_nomemory;
			throw BinarySequenceBadAlloc();
			}
		catch(...){
			error=binseq_unknown;
			throw BinarySequenceError();
		}
	
return seq.size();
}

// .............................................................................
// Name: push
//
// Synopsis: Adds a value to the end of the sequence
//
// Parameters: 
//			 char c    -----> The added value. if c is odd a 1, 0 otherwise
//		
// Returns: 
//         std::vector<bool>::size_type ------> the size of the sequence
//   
// Exceptions: 
//            BinarySequenceBadAlloc      --------> Memory allocation failed. error flag is set.
//            BinarySequenceError         --------> Generic error. error flag is set.
//..............................................................................
inline std::vector<bool>::size_type binsequence::push(char c){
	
	try {
		seq.push_back((bool)(c%2));
		}
	catch (std::bad_alloc & ba){
			error=binseq_nomemory;
			throw BinarySequenceBadAlloc();
			}
		catch(...){
			error=binseq_unknown;
			throw BinarySequenceError();
		}
	
return seq.size();
}

// .............................................................................
// Name: push
//
// Synopsis: Adds a value to the end of the sequence
//
// Parameters: 
//			 int c    -----> The added value. if c is odd a 1, 0 otherwise
//		
// Returns: 
//         std::vector<bool>::size_type ------> the size of the sequence
//   
// Exceptions: 
//            BinarySequenceBadAlloc      --------> Memory allocation failed. error flag is set.
//            BinarySequenceError         --------> Generic error. error flag is set.
//..............................................................................
inline std::vector<bool>::size_type binsequence::push(int c){
	
	try {
		seq.push_back((bool)(c%2));
		}
	catch (std::bad_alloc & ba){
			error=binseq_nomemory;
			throw BinarySequenceBadAlloc();
			}
		catch(...){
			error=binseq_unknown;
			throw BinarySequenceError();
		}	
	
return seq.size();
}

// .............................................................................
// Name: pop
//
// Synopsis: erase the last value of the sequence
//
// Parameters: 
//			 None
//		
// Returns: 
//         std::vector<bool>::size_type ------> the size of the sequence
//   
// Exceptions: 
//            None
//..............................................................................
inline std::vector<bool>::size_type binsequence::pop(void){
	seq.pop_back();
	
return seq.size();
}

// .............................................................................
// Name: clear
//
// Synopsis: erase the whole sequence
//
// Parameters: 
//			 None
//		
// Returns: 
//         None
//   
// Exceptions: 
//            None
//..............................................................................
inline void binsequence::clear(void){
	error=binseq_ok;
	seq.clear();
}

// .............................................................................
// Name: pi
//
// Synopsis: erase the last value of the sequence
//
// Parameters: 
//			 None
//		
// Returns: 
//         std::vector<bool>::size_type ------> the size of the sequence
//   
// Exceptions: 
//            None
//..............................................................................
inline binsequence & binsequence::pi(void){
	seq.pop_back();
return *this;	
}

// .............................................................................
// Name: reverse
//
// Synopsis: Reverses the seqeunce
//
// Parameters: 
//			 None
//		
// Returns: 
//         *this
//   
// Exceptions: 
//            None
//..............................................................................
inline binsequence & binsequence::reverse(void){
	std::reverse(seq.begin(),seq.end());
	
return *this;
}

// .............................................................................
// Name: negate
//
// Synopsis: Negates (flips) the seqeunce
//
// Parameters: 
//			 None
//		
// Returns: 
//         *this
//   
// Exceptions: 
//            None
//..............................................................................
inline binsequence & binsequence::negate(void){
	seq.flip();
	
return *this;
}

// .............................................................................
// Name: rightshift
//
// Synopsis: Circular right shift of the sequence
//
// Parameters: 
//			 std::vector<bool>::size_type ls  --> Number of positions to shift
//		
// Returns: 
//         *this
//   
// Exceptions: 
//            None
//..............................................................................		
inline binsequence  & binsequence::rightshift(std::vector<bool>::size_type ls){
	std::rotate(seq.begin(), seq.begin()+(ls%seq.size()), seq.end());
	
return *this;	
}

// .............................................................................
// Name: leftshift
//
// Synopsis: Circular left shift of the sequence
//
// Parameters: 
//			 std::vector<bool>::size_type ls  --> Number of positions to shift
//		
// Returns: 
//         *this
//   
// Exceptions: 
//            None
//..............................................................................
inline binsequence  & binsequence::leftshift(std::vector<bool>::size_type ls){
	std::rotate(seq.begin(), seq.begin()+seq.size()-(ls%seq.size()), seq.end());
	
return *this;	
}

// .............................................................................
// Name: size
//
// Synopsis: The sequence length
//
// Parameters: 
//			 None
//		
// Returns: 
//         std::vector<char>::size_type ---> The sequence length
//   
// Exceptions: 
//            None
//..............................................................................
inline std::vector<char>::size_type binsequence::size(void)const{
	return seq.size();
}

// .............................................................................
// Name: flip
//
// Synopsis: Flips a sequence value
//
// Parameters: 
//			 std::vector<bool>::size_type index  --> the position of the value to be flipped
//		
// Returns: 
//         bool   ---> The new value
//   
// Exceptions: 
//            BinaryOutOfBounds           --------> Index is out of bounds. error flag is set.
//..............................................................................
inline bool binsequence::flip(std::vector<bool>::size_type index){
std::vector<bool>::size_type i=index;

	error=binseq_ok;
	
	if(index >= seq.size()){
	    i=seq.size()-1;
	    error=binseq_outofbounds;
	    throw BinarySequenceOutOfBounds(); 
		}
	else 
		seq[i]=(seq[i]==true)? false : true;
		
return seq[i];	
}

// .............................................................................
// Name: bit
//
// Synopsis: Test the sequence value
//
// Parameters: 
//			 std::vector<bool>::size_type index  --> the position of the value to be flipped
//		
// Returns: 
//         bool   ---> The bit value
//   
// Exceptions: 
//            BinaryOutOfBounds           --------> Index is out of bounds. error flag is set.
//..............................................................................
inline bool binsequence::bit(std::vector<bool>::size_type index)const{
std::vector<bool>::size_type i=index;
	
	if(index >= seq.size()){
	    i=seq.size()-1;
	    throw BinarySequenceOutOfBounds(); 
		}
		
return seq[i];	
}

// .............................................................................
// Name: set
//
// Synopsis: Set to one a sequence value
//
// Parameters: 
//			 std::vector<bool>::size_type index  --> the position of the value to be set
//		
// Returns: 
//         None
//   
// Exceptions: 
//            BinaryOutOfBounds           --------> Index is out of bounds. error flag is set.
//..............................................................................
inline void binsequence::set(std::vector<bool>::size_type index){
	std::vector<bool>::size_type i=index;
	error=binseq_ok;
	
	if(index >= seq.size()){
	    i=seq.size()-1;
	    error=binseq_outofbounds;
	    throw BinarySequenceOutOfBounds(); 
		}
	else 
		seq[i]=true;
}   
    
// .............................................................................
// Name: unset
//
// Synopsis: Set to zero a sequence value
//
// Parameters: 
//			 std::vector<bool>::size_type index  --> the position of the value to be unset
//		
// Returns: 
//         None
//   
// Exceptions: 
//            BinaryOutOfBounds           --------> Index is out of bounds. error flag is set.
//..............................................................................                                          
inline void binsequence::unset(std::vector<bool>::size_type index){
	std::vector<bool>::size_type i=index;
	error=binseq_ok;
	
	if(index >= seq.size()){
	    i=seq.size()-1;
	    error=binseq_outofbounds;
	    throw BinarySequenceOutOfBounds(); 
		}
	else 
		seq[i]=false;
} 



// .............................................................................
// Name: NoZero
//
// Synopsis: Calculates the number of zero in the sequence 
//
// Parameters: 
//			None
//		
// Returns: 
//         The number of zero as an unsigned long
//   
// Exceptions: 
//            None
//..............................................................................       
inline unsigned long binsequence::NoZero(void)const{
unsigned long result=0;

	for(bool i: seq)
	   result+=((i==false)? 1: 0);

return result;
}

// .............................................................................
// Name: NoOnes
//
// Synopsis: Calculates the number of ones in the sequence 
//
// Parameters: 
//			None
//		
// Returns: 
//         The number of ones as an unsigned long
//   
// Exceptions: 
//            None
//.............................................................................. 
inline unsigned long binsequence::NoOnes(void)const{
return seq.size()-NoZero();
}  


// .............................................................................
// Name: Search
//
// Synopsis: the index of the first starting position in the sequence
//
// Parameters: 
//			const binsequence & ss    -----> Sequence to search for
//          unsigned long start=0     -----> The starting position to start the search
//		
// Returns: 
//         The index of the starting position of the first occurence, after start, of sequence ss within the object
//   
// Exceptions: 
//            None
//.............................................................................. 
inline unsigned long binsequence::Search(const binsequence & ss, unsigned long start=0)const{
	
	return std::search(seq.begin()+start, seq.end(), ss.seq.begin(), ss.seq.end())-seq.begin();
}       

// .......................................................
//                friend functions
//........................................................ 

// .............................................................................
// Name: swap
//
// Synopsis: swap operator
//
// Parameters: 
//			binsequence & first     -----> first factor
//          binsequence & second    -----> second factor
//	
// Returns: 
//         None
//   
// Exceptions: 
//            BinarySequenceBadAlloc      --------> Memory allocation failed. error flag is set.
//            BinarySequenceError         --------> Generic error. error flag is set.
//.............................................................................. 
 inline void swap(binsequence & first, binsequence & second){
	try{
		std::swap(first.error, second.error); 
		std::swap(first.seq, second.seq);
		}
   	catch (std::bad_alloc & ba){
			throw BinarySequenceBadAlloc();
			}
		catch(...){
			throw BinarySequenceError();
		}
 }
 
 


// .............................................................................
// Name: for_each
//
// Synopsis: map operator
//
// Parameters: 
//			std::function<bool(bool)>  fn   -----> the mapping function
//          binsequence & s                 -----> the sequence mapping is performed to 
//	
// Returns: 
//         The transformed sequence
//   
// Exceptions: 
//            BinarySequenceBadAlloc      --------> Memory allocation failed. error flag not set.
//            BinarySequenceError         --------> Generic error. error flag not set.
//..............................................................................  
 inline binsequence for_each(std::function<bool(bool)> fn, const binsequence & s){
	return map(fn,s);
 }
 
// .............................................................................
// Name: operator==
//
// Synopsis: logical equality operator
//
// Parameters: 
//			const binsequence & lhs   -----> First factor
//          const binsequence & rhs   -----> Second factor
//	
// Returns: 
//         true if sequences are equal; false otherwise
//   
// Exceptions: 
//            None
//.............................................................................. 
inline bool operator==(const binsequence & lhs, const binsequence & rhs){
	return lhs.seq==rhs.seq;
}

// .............................................................................
// Name: operator!=
//
// Synopsis: logical non-equal operator
//
// Parameters: 
//			const binsequence & lhs   -----> First factor
//          const binsequence & rhs   -----> Second factor
//	
// Returns: 
//         true if sequences are not equal; false otherwise
//   
// Exceptions: 
//            None
//..............................................................................
inline bool operator!=(const binsequence & lhs, const binsequence & rhs){
	return !operator==(lhs,rhs);
}

// .............................................................................
// Name: operator<
//
// Synopsis: lexicographically smaller than operator
//
// Parameters: 
//			const binsequence & lhs   -----> First factor
//          const binsequence & rhs   -----> Second factor
//	
// Returns: 
//         true if lhs is smaller than rhs; false otherwise
//   
// Exceptions: 
//            None
//..............................................................................
inline bool operator<(const binsequence & lhs, const binsequence & rhs){
	return lhs.seq < rhs.seq;
}

// .............................................................................
// Name: operator<=
//
// Synopsis: lexicographically smaller or equal than operator
//
// Parameters: 
//			const binsequence & lhs   -----> First factor
//          const binsequence & rhs   -----> Second factor
//	
// Returns: 
//         true if lhs is smaller or equal than rhs; false otherwise
//   
// Exceptions: 
//            None
//..............................................................................		
inline bool operator<=(const binsequence & lhs, const binsequence & rhs){
	return !operator> (lhs,rhs);
}

// .............................................................................
// Name: operator>
//
// Synopsis: lexicographically larger than operator
//
// Parameters: 
//			const binsequence & lhs   -----> First factor
//          const binsequence & rhs   -----> Second factor
//	
// Returns: 
//         true if lhs is larger than rhs; false otherwise
//   
// Exceptions: 
//            None
//..............................................................................		
inline bool operator>(const binsequence & lhs, const binsequence & rhs){
	return  operator< (rhs,lhs);
}

// .............................................................................
// Name: operator>=
//
// Synopsis: lexicographically larger or equal than operator
//
// Parameters: 
//			const binsequence & lhs   -----> First factor
//          const binsequence & rhs   -----> Second factor
//	
// Returns: 
//         true if lhs is larger or equal than rhs; false otherwise
//   
// Exceptions: 
//            None
//..............................................................................
inline bool operator>=(const binsequence &lhs , const binsequence & rhs){
	return !operator< (lhs,rhs);
}

// .............................................................................
// Name: operator&
//
// Synopsis: bitwise "and" operator
//
// Parameters: 
//			const binsequence & first  -----> first factor
//          const binsequence & second -----> second factor 
//         
// Returns: 
//         the bitwise "and" sequence
//   
// Exceptions: 
//            BinarySequenceBadAlloc      --------> Memory allocation failed. error flag not set.
//            BinarySequenceError         --------> Generic error. error flag not set.
//.............................................................................. 
inline binsequence operator&(const binsequence & first, binsequence &second){
	
	binsequence result(first);
	
	return result&=second;
}

// .............................................................................
// Name: operator|
//
// Synopsis: bitwise "or" operator
//
// Parameters: 
//			const binsequence & first  -----> first factor
//          const binsequence & second -----> second factor 
//         
// Returns: 
//         the bitwise "or" sequence
//   
// Exceptions: 
//            BinarySequenceBadAlloc      --------> Memory allocation failed. error flag not set.
//            BinarySequenceError         --------> Generic error. error flag not set.
//.............................................................................. 
inline binsequence operator|(const binsequence & first, binsequence &second){

	binsequence result(first);
	
	return result|=second;
}

// .............................................................................
// Name: operator^
//
// Synopsis: bitwise "xor" operator. Hamming field.
//
// Parameters: 
//			const binsequence & first  -----> first factor
//          const binsequence & second -----> second factor 
//         
// Returns: 
//         the Hamming field
//   
// Exceptions: 
//            BinarySequenceBadAlloc      --------> Memory allocation failed. error flag not set.
//            BinarySequenceError         --------> Generic error. error flag not set.
//.............................................................................. 
inline binsequence operator^(const binsequence & first, binsequence &second){

	binsequence result(first);
	
	return result^=second;
}

// .............................................................................
// Name: operator+
//
// Synopsis: bitwise "or" operator. 
//
// Parameters: 
//			const binsequence & first  -----> first factor
//          const binsequence & second -----> second factor 
//         
// Returns: 
//         the "or" sequence
//   
// Exceptions: 
//            BinarySequenceBadAlloc      --------> Memory allocation failed. error flag not set.
//            BinarySequenceError         --------> Generic error. error flag not set.
//.............................................................................. 
inline binsequence operator+(const binsequence & first, binsequence &second){
	binsequence result(first);
	
	return result+=second;
	
}

// .............................................................................
// Name: operator-
//
// Synopsis: bitwise "xor" operator. Hamming field.
//
// Parameters: 
//			const binsequence & first  -----> first factor
//          const binsequence & second -----> second factor 
//         
// Returns: 
//         the sequence corresponding to the Hamming field
//   
// Exceptions: 
//            BinarySequenceBadAlloc      --------> Memory allocation failed. error flag not set.
//            BinarySequenceError         --------> Generic error. error flag not set.
//.............................................................................. 
inline binsequence operator-(const binsequence & first, binsequence &second){
	binsequence result(first);
	
	return result-=second;
	
}

// internal auxiliary function
inline bool bsnot(bool f){
	return !f;
}

// .............................................................................
// Name: operator!
//
// Synopsis: bitwise "not" operator. 
//
// Parameters: 
//			const binsequence & s  -----> operand
//         
// Returns: 
//         the negated sequence
//   
// Exceptions: 
//            BinarySequenceBadAlloc      --------> Memory allocation failed. error flag not set.
//            BinarySequenceError         --------> Generic error. error flag not set.
//.............................................................................. 
inline binsequence operator!(const binsequence & s){
binsequence result(s);


std::for_each(result.seq.begin(), result.seq.end(), bsnot );
	
return result;	
}

//...........................................................
//                      Free functions
//...........................................................
 
#endif
