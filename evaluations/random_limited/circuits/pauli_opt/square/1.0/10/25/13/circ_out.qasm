OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
cx q[9],q[10];
cx q[13],q[14];
cx q[6],q[5];
cx q[7],q[2];
cx q[11],q[18];
cx q[24],q[15];
cx q[2],q[1];
cx q[21],q[22];
cx q[18],q[19];
cx q[7],q[6];
cx q[10],q[9];
cx q[9],q[10];
cx q[5],q[4];
cx q[7],q[12];
cx q[15],q[14];
cx q[7],q[6];
rz(pi) q[6];
cx q[7],q[6];
cx q[16],q[13];
rz(pi) q[13];
cx q[16],q[13];
cx q[18],q[11];
rz(pi) q[11];
cx q[18],q[11];
rz(pi) q[20];
cx q[1],q[2];
cx q[0],q[1];
rx(pi) q[0];
cx q[0],q[1];
cx q[1],q[2];
rx(pi) q[9];
cx q[11],q[18];
rx(pi) q[11];
cx q[11],q[18];
rz(7*pi/4) q[23];
rz(7*pi/4) q[0];
cx q[22],q[12];
rz(3*pi/4) q[12];
cx q[22],q[12];
cx q[6],q[7];
rx(3*pi/2) q[6];
cx q[6],q[7];
cx q[18],q[17];
cx q[11],q[18];
cx q[10],q[11];
rx(3*pi/2) q[10];
cx q[10],q[11];
cx q[11],q[18];
cx q[18],q[17];
cx q[17],q[18];
cx q[16],q[17];
rx(3*pi/4) q[16];
cx q[16],q[17];
cx q[17],q[18];
cx q[14],q[15];
rx(pi/2) q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[10];
rz(5*pi/4) q[10];
cx q[12],q[10];
cx q[13],q[12];
cx q[14],q[13];
cx q[15],q[14];
cx q[7],q[6];
rz(pi) q[6];
cx q[7],q[6];
cx q[16],q[13];
rz(pi) q[13];
cx q[16],q[13];
cx q[18],q[11];
rz(pi) q[11];
cx q[18],q[11];
rz(pi) q[20];
cx q[1],q[2];
cx q[0],q[1];
rx(pi) q[0];
cx q[0],q[1];
cx q[1],q[2];
rx(pi) q[9];
cx q[11],q[18];
rx(pi) q[11];
cx q[11],q[18];
rz(7*pi/4) q[23];
rz(7*pi/4) q[0];
cx q[22],q[12];
rz(3*pi/4) q[12];
cx q[22],q[12];
cx q[6],q[7];
rx(3*pi/2) q[6];
cx q[6],q[7];
cx q[18],q[17];
cx q[11],q[18];
cx q[10],q[11];
rx(3*pi/2) q[10];
cx q[10],q[11];
cx q[11],q[18];
cx q[18],q[17];
cx q[17],q[18];
cx q[16],q[17];
rx(3*pi/4) q[16];
cx q[16],q[17];
cx q[17],q[18];
cx q[14],q[15];
rx(pi/2) q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[10];
rz(5*pi/4) q[10];
cx q[12],q[10];
cx q[13],q[12];
cx q[14],q[13];
cx q[15],q[14];
cx q[7],q[6];
rz(pi) q[6];
cx q[7],q[6];
cx q[16],q[13];
rz(pi) q[13];
cx q[16],q[13];
cx q[18],q[11];
rz(pi) q[11];
cx q[18],q[11];
rz(pi) q[20];
cx q[1],q[2];
cx q[0],q[1];
rx(pi) q[0];
cx q[0],q[1];
cx q[1],q[2];
rx(pi) q[9];
cx q[11],q[18];
rx(pi) q[11];
cx q[11],q[18];
rz(7*pi/4) q[23];
rz(7*pi/4) q[0];
cx q[22],q[12];
rz(3*pi/4) q[12];
cx q[22],q[12];
cx q[6],q[7];
rx(3*pi/2) q[6];
cx q[6],q[7];
cx q[18],q[17];
cx q[11],q[18];
cx q[10],q[11];
rx(3*pi/2) q[10];
cx q[10],q[11];
cx q[11],q[18];
cx q[18],q[17];
cx q[17],q[18];
cx q[16],q[17];
rx(3*pi/4) q[16];
cx q[16],q[17];
cx q[17],q[18];
cx q[14],q[15];
rx(pi/2) q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[10];
rz(5*pi/4) q[10];
cx q[12],q[10];
cx q[13],q[12];
cx q[14],q[13];
cx q[15],q[14];
cx q[7],q[6];
rz(pi) q[6];
cx q[7],q[6];
cx q[16],q[13];
rz(pi) q[13];
cx q[16],q[13];
cx q[18],q[11];
rz(pi) q[11];
cx q[18],q[11];
rz(pi) q[20];
cx q[1],q[2];
cx q[0],q[1];
rx(pi) q[0];
cx q[0],q[1];
cx q[1],q[2];
rx(pi) q[9];
cx q[11],q[18];
rx(pi) q[11];
cx q[11],q[18];
rz(7*pi/4) q[23];
rz(7*pi/4) q[0];
cx q[22],q[12];
rz(3*pi/4) q[12];
cx q[22],q[12];
cx q[6],q[7];
rx(3*pi/2) q[6];
cx q[6],q[7];
cx q[18],q[17];
cx q[11],q[18];
cx q[10],q[11];
rx(3*pi/2) q[10];
cx q[10],q[11];
cx q[11],q[18];
cx q[18],q[17];
cx q[17],q[18];
cx q[16],q[17];
rx(3*pi/4) q[16];
cx q[16],q[17];
cx q[17],q[18];
cx q[14],q[15];
rx(pi/2) q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[10];
rz(5*pi/4) q[10];
cx q[12],q[10];
cx q[13],q[12];
cx q[14],q[13];
cx q[15],q[14];
cx q[7],q[6];
rz(pi) q[6];
cx q[7],q[6];
cx q[16],q[13];
rz(pi) q[13];
cx q[16],q[13];
cx q[18],q[11];
rz(pi) q[11];
cx q[18],q[11];
rz(pi) q[20];
cx q[1],q[2];
cx q[0],q[1];
rx(pi) q[0];
cx q[0],q[1];
cx q[1],q[2];
rx(pi) q[9];
cx q[11],q[18];
rx(pi) q[11];
cx q[11],q[18];
rz(7*pi/4) q[23];
rz(7*pi/4) q[0];
cx q[22],q[12];
rz(3*pi/4) q[12];
cx q[22],q[12];
cx q[6],q[7];
rx(3*pi/2) q[6];
cx q[6],q[7];
cx q[18],q[17];
cx q[11],q[18];
cx q[10],q[11];
rx(3*pi/2) q[10];
cx q[10],q[11];
cx q[11],q[18];
cx q[18],q[17];
cx q[17],q[18];
cx q[16],q[17];
rx(3*pi/4) q[16];
cx q[16],q[17];
cx q[17],q[18];
cx q[14],q[15];
rx(pi/2) q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[10];
rz(5*pi/4) q[10];
cx q[12],q[10];
cx q[13],q[12];
cx q[14],q[13];
cx q[15],q[14];
cx q[9],q[10];
cx q[5],q[4];
cx q[7],q[12];
cx q[15],q[14];
cx q[2],q[1];
cx q[21],q[22];
cx q[18],q[19];
cx q[7],q[6];
cx q[10],q[9];
cx q[9],q[10];
cx q[13],q[14];
cx q[6],q[5];
cx q[7],q[2];
cx q[11],q[18];
cx q[24],q[15];
