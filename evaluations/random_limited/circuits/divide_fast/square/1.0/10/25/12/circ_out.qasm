OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
cx q[3],q[2];
cx q[4],q[3];
cx q[3],q[2];
cx q[4],q[3];
cx q[8],q[11];
rz(pi) q[0];
rz(pi) q[5];
rz(pi) q[17];
rz(pi) q[23];
rx(pi) q[3];
rx(pi) q[12];
rz(5*pi/4) q[2];
rz(pi/2) q[7];
rx(3*pi/4) q[22];
cx q[20],q[21];
cx q[21],q[20];
cx q[21],q[22];
cx q[22],q[21];
cx q[23],q[22];
cx q[23],q[16];
cx q[16],q[23];
cx q[16],q[13];
cx q[13],q[16];
cx q[14],q[13];
rx(5*pi/4) q[14];
cx q[14],q[13];
cx q[13],q[16];
cx q[16],q[13];
cx q[16],q[23];
cx q[23],q[16];
cx q[23],q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[21],q[18];
cx q[18],q[21];
cx q[11],q[18];
cx q[11],q[12];
cx q[8],q[11];
rx(5*pi/4) q[8];
cx q[8],q[11];
cx q[11],q[12];
cx q[11],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[21],q[20];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[11];
rz(pi/2) q[11];
cx q[12],q[11];
cx q[13],q[12];
cx q[12],q[13];
cx q[18],q[19];
cx q[18],q[17];
cx q[17],q[18];
cx q[16],q[17];
rx(pi/4) q[16];
cx q[16],q[17];
cx q[17],q[18];
cx q[18],q[17];
cx q[18],q[19];
cx q[17],q[22];
cx q[22],q[17];
cx q[12],q[17];
cx q[17],q[12];
cx q[7],q[12];
cx q[12],q[7];
cx q[7],q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(3*pi/2) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[7],q[2];
cx q[12],q[7];
cx q[7],q[12];
cx q[17],q[12];
cx q[12],q[17];
cx q[22],q[17];
cx q[17],q[22];
rz(pi) q[0];
rz(pi) q[5];
rz(pi) q[17];
rz(pi) q[23];
rx(pi) q[3];
rx(pi) q[12];
rz(5*pi/4) q[2];
rz(pi/2) q[7];
rx(3*pi/4) q[22];
cx q[20],q[21];
cx q[21],q[20];
cx q[21],q[22];
cx q[22],q[21];
cx q[23],q[22];
cx q[23],q[16];
cx q[16],q[23];
cx q[16],q[13];
cx q[13],q[16];
cx q[14],q[13];
rx(5*pi/4) q[14];
cx q[14],q[13];
cx q[13],q[16];
cx q[16],q[13];
cx q[16],q[23];
cx q[23],q[16];
cx q[23],q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[21],q[18];
cx q[18],q[21];
cx q[11],q[18];
cx q[11],q[12];
cx q[8],q[11];
rx(5*pi/4) q[8];
cx q[8],q[11];
cx q[11],q[12];
cx q[11],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[21],q[20];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[11];
rz(pi/2) q[11];
cx q[12],q[11];
cx q[13],q[12];
cx q[12],q[13];
cx q[18],q[19];
cx q[18],q[17];
cx q[17],q[18];
cx q[16],q[17];
rx(pi/4) q[16];
cx q[16],q[17];
cx q[17],q[18];
cx q[18],q[17];
cx q[18],q[19];
cx q[17],q[22];
cx q[22],q[17];
cx q[12],q[17];
cx q[17],q[12];
cx q[7],q[12];
cx q[12],q[7];
cx q[7],q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(3*pi/2) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[7],q[2];
cx q[12],q[7];
cx q[7],q[12];
cx q[17],q[12];
cx q[12],q[17];
cx q[22],q[17];
cx q[17],q[22];
rz(pi) q[0];
rz(pi) q[5];
rz(pi) q[17];
rz(pi) q[23];
rx(pi) q[3];
rx(pi) q[12];
rz(5*pi/4) q[2];
rz(pi/2) q[7];
rx(3*pi/4) q[22];
cx q[20],q[21];
cx q[21],q[20];
cx q[21],q[22];
cx q[22],q[21];
cx q[23],q[22];
cx q[23],q[16];
cx q[16],q[23];
cx q[16],q[13];
cx q[13],q[16];
cx q[14],q[13];
rx(5*pi/4) q[14];
cx q[14],q[13];
cx q[13],q[16];
cx q[16],q[13];
cx q[16],q[23];
cx q[23],q[16];
cx q[23],q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[21],q[18];
cx q[18],q[21];
cx q[11],q[18];
cx q[11],q[12];
cx q[8],q[11];
rx(5*pi/4) q[8];
cx q[8],q[11];
cx q[11],q[12];
cx q[11],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[21],q[20];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[11];
rz(pi/2) q[11];
cx q[12],q[11];
cx q[13],q[12];
cx q[12],q[13];
cx q[18],q[19];
cx q[18],q[17];
cx q[17],q[18];
cx q[16],q[17];
rx(pi/4) q[16];
cx q[16],q[17];
cx q[17],q[18];
cx q[18],q[17];
cx q[18],q[19];
cx q[17],q[22];
cx q[22],q[17];
cx q[12],q[17];
cx q[17],q[12];
cx q[7],q[12];
cx q[12],q[7];
cx q[7],q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(3*pi/2) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[7],q[2];
cx q[12],q[7];
cx q[7],q[12];
cx q[17],q[12];
cx q[12],q[17];
cx q[22],q[17];
cx q[17],q[22];
rz(pi) q[0];
rz(pi) q[5];
rz(pi) q[17];
rz(pi) q[23];
rx(pi) q[3];
rx(pi) q[12];
rz(5*pi/4) q[2];
rz(pi/2) q[7];
rx(3*pi/4) q[22];
cx q[20],q[21];
cx q[21],q[20];
cx q[21],q[22];
cx q[22],q[21];
cx q[23],q[22];
cx q[23],q[16];
cx q[16],q[23];
cx q[16],q[13];
cx q[13],q[16];
cx q[14],q[13];
rx(5*pi/4) q[14];
cx q[14],q[13];
cx q[13],q[16];
cx q[16],q[13];
cx q[16],q[23];
cx q[23],q[16];
cx q[23],q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[21],q[18];
cx q[18],q[21];
cx q[11],q[18];
cx q[11],q[12];
cx q[8],q[11];
rx(5*pi/4) q[8];
cx q[8],q[11];
cx q[11],q[12];
cx q[11],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[21],q[20];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[11];
rz(pi/2) q[11];
cx q[12],q[11];
cx q[13],q[12];
cx q[12],q[13];
cx q[18],q[19];
cx q[18],q[17];
cx q[17],q[18];
cx q[16],q[17];
rx(pi/4) q[16];
cx q[16],q[17];
cx q[17],q[18];
cx q[18],q[17];
cx q[18],q[19];
cx q[17],q[22];
cx q[22],q[17];
cx q[12],q[17];
cx q[17],q[12];
cx q[7],q[12];
cx q[12],q[7];
cx q[7],q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(3*pi/2) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[7],q[2];
cx q[12],q[7];
cx q[7],q[12];
cx q[17],q[12];
cx q[12],q[17];
cx q[22],q[17];
cx q[17],q[22];
rz(pi) q[0];
rz(pi) q[5];
rz(pi) q[17];
rz(pi) q[23];
rx(pi) q[3];
rx(pi) q[12];
rz(5*pi/4) q[2];
rz(pi/2) q[7];
rx(3*pi/4) q[22];
cx q[20],q[21];
cx q[21],q[20];
cx q[21],q[22];
cx q[22],q[21];
cx q[23],q[22];
cx q[23],q[16];
cx q[16],q[23];
cx q[16],q[13];
cx q[13],q[16];
cx q[14],q[13];
rx(5*pi/4) q[14];
cx q[14],q[13];
cx q[13],q[16];
cx q[16],q[13];
cx q[16],q[23];
cx q[23],q[16];
cx q[23],q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[21],q[18];
cx q[18],q[21];
cx q[11],q[18];
cx q[11],q[12];
cx q[8],q[11];
rx(5*pi/4) q[8];
cx q[8],q[11];
cx q[11],q[12];
cx q[11],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[21],q[20];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[11];
rz(pi/2) q[11];
cx q[12],q[11];
cx q[13],q[12];
cx q[12],q[13];
cx q[18],q[19];
cx q[18],q[17];
cx q[17],q[18];
cx q[16],q[17];
rx(pi/4) q[16];
cx q[16],q[17];
cx q[17],q[18];
cx q[18],q[17];
cx q[18],q[19];
cx q[17],q[22];
cx q[22],q[17];
cx q[12],q[17];
cx q[17],q[12];
cx q[7],q[12];
cx q[12],q[7];
cx q[7],q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(3*pi/2) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[7],q[2];
cx q[12],q[7];
cx q[7],q[12];
cx q[17],q[12];
cx q[12],q[17];
cx q[22],q[17];
cx q[17],q[22];
cx q[3],q[2];
cx q[4],q[3];
cx q[3],q[2];
cx q[4],q[3];
cx q[8],q[11];
