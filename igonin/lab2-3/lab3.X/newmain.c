#include <xc.h>
#include <stdlib.h>
#include <stdio.h>
#include "config.h"
#include <stdbool.h>
#include <time.h>

int cnt = 0;

void interrupt isr(){
    if(INTF){
        INTF = 0; // reset interrupt flag/
        RB6 = cnt & 1;
        RB5 = (cnt >> 1) & 1;
        RA6 = (cnt >> 2) & 1;
        RA7 = (cnt >> 3) & 1;
        RB7 = 1;
        RB1 = 1;
        __delay_ms(1000);
        RB7 = 0;
    }
    if(RBIF){
        if(RB7){
            for(int i = 0; i < 20; ++i){
                RB2 = ~RB2;
                __delay_ms(100);
            }
        }
        RBIF = 0;
    }
}

void main(void) {
    TRISA = 0b00001111;
    TRISB = 0B10000001;
    PORTA = 0;
    PORTB = 0;

    INTF = 0;    //reset the external interrupt flag
    INTEDG = 1;  //interrupt on the rising edge
    INTE = 1;    //enable the external interrupt
    GIE = 1;     //set the Global Interrupt Enable
    RBIE = 1;
    RBIF = 0;
    
    while(1)
    {
        __delay_ms(1000);
        ++cnt;
        bool I1 = RA0;
        bool I2 = RA1;
        bool I3 = RA2;
        bool I4 = RA3;
        bool I5 = RA4;

        bool T1 = (cnt % 2 == 0);
        bool T2 = (cnt % 5 == 0);
        bool T3 = (cnt % 7 == 0);
        
        bool O1 = ((I1 && I2 && I3) || (I3 && I4 && I5)) && T2;
        bool O2 = !O1;
        bool O3 = (!I1 || !I2 || !I3 || !I4) && T3;
        bool O4 = (I1 && T1) || (I2 && T2) || (I3 && T3);
        RB1 = O1;
        RB2 = O2;
        RB3 = O3;
        RB4 = O4;

    }
    return;
}
