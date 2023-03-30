OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
cx q[13],q[14];
cx q[11],q[10];
cx q[22],q[23];
cx q[7],q[6];
cx q[1],q[0];
cx q[3],q[2];
cx q[16],q[17];
cx q[20],q[21];
cx q[17],q[22];
cx q[16],q[15];
cx q[13],q[6];
cx q[9],q[8];
cx q[14],q[5];
cx q[5],q[14];
cx q[20],q[21];
cx q[12],q[7];
cx q[22],q[23];
rz(pi) q[8];
rz(pi) q[19];
cx q[23],q[16];
cx q[17],q[16];
rz(pi) q[16];
cx q[17],q[16];
cx q[23],q[16];
rx(pi) q[21];
cx q[22],q[17];
rz(3*pi/2) q[17];
cx q[22],q[17];
rz(5*pi/4) q[14];
rz(3*pi/4) q[2];
cx q[8],q[7];
cx q[12],q[7];
rz(5*pi/4) q[7];
cx q[12],q[7];
cx q[8],q[7];
rz(3*pi/2) q[4];
rz(pi/2) q[19];
cx q[6],q[11];
rx(3*pi/2) q[6];
cx q[6],q[11];
cx q[7],q[12];
cx q[1],q[7];
rx(pi/2) q[1];
cx q[1],q[7];
cx q[7],q[12];
cx q[5],q[9];
rx(5*pi/4) q[5];
cx q[5],q[9];
cx q[10],q[20];
rx(3*pi/4) q[10];
cx q[10],q[20];
cx q[3],q[23];
cx q[2],q[3];
rx(pi/4) q[2];
cx q[2],q[3];
cx q[3],q[23];
rx(5*pi/4) q[5];
rx(7*pi/4) q[22];
rx(5*pi/4) q[15];
cx q[16],q[13];
cx q[13],q[6];
rz(pi/4) q[6];
cx q[13],q[6];
cx q[16],q[13];
rz(5*pi/4) q[11];
rz(7*pi/4) q[3];
rz(pi/4) q[12];
cx q[19],q[9];
rz(5*pi/4) q[9];
cx q[19],q[9];
cx q[23],q[16];
cx q[16],q[17];
cx q[17],q[8];
rz(pi/2) q[8];
cx q[17],q[8];
cx q[16],q[17];
cx q[23],q[16];
cx q[17],q[16];
cx q[16],q[13];
cx q[13],q[14];
cx q[14],q[5];
rz(pi/4) q[5];
cx q[14],q[5];
cx q[13],q[14];
cx q[16],q[13];
cx q[17],q[16];
cx q[17],q[21];
cx q[16],q[17];
rx(3*pi/4) q[16];
cx q[16],q[17];
cx q[17],q[21];
cx q[7],q[12];
cx q[2],q[7];
cx q[2],q[3];
rx(pi/4) q[2];
cx q[2],q[3];
cx q[2],q[7];
cx q[7],q[12];
cx q[11],q[21];
rx(3*pi/4) q[11];
cx q[11],q[21];
cx q[13],q[12];
rz(pi/2) q[12];
cx q[13],q[12];
rz(3*pi/2) q[3];
rz(3*pi/4) q[16];
rz(pi) q[8];
rz(pi) q[19];
cx q[23],q[16];
cx q[17],q[16];
rz(pi) q[16];
cx q[17],q[16];
cx q[23],q[16];
rx(pi) q[21];
cx q[22],q[17];
rz(3*pi/2) q[17];
cx q[22],q[17];
rz(5*pi/4) q[14];
rz(3*pi/4) q[2];
cx q[8],q[7];
cx q[12],q[7];
rz(5*pi/4) q[7];
cx q[12],q[7];
cx q[8],q[7];
rz(3*pi/2) q[4];
rz(pi/2) q[19];
cx q[6],q[11];
rx(3*pi/2) q[6];
cx q[6],q[11];
cx q[7],q[12];
cx q[1],q[7];
rx(pi/2) q[1];
cx q[1],q[7];
cx q[7],q[12];
cx q[5],q[9];
rx(5*pi/4) q[5];
cx q[5],q[9];
cx q[10],q[20];
rx(3*pi/4) q[10];
cx q[10],q[20];
cx q[3],q[23];
cx q[2],q[3];
rx(pi/4) q[2];
cx q[2],q[3];
cx q[3],q[23];
rx(5*pi/4) q[5];
rx(7*pi/4) q[22];
rx(5*pi/4) q[15];
cx q[16],q[13];
cx q[13],q[6];
rz(pi/4) q[6];
cx q[13],q[6];
cx q[16],q[13];
rz(5*pi/4) q[11];
rz(7*pi/4) q[3];
rz(pi/4) q[12];
cx q[19],q[9];
rz(5*pi/4) q[9];
cx q[19],q[9];
cx q[23],q[16];
cx q[16],q[17];
cx q[17],q[8];
rz(pi/2) q[8];
cx q[17],q[8];
cx q[16],q[17];
cx q[23],q[16];
cx q[17],q[16];
cx q[16],q[13];
cx q[13],q[14];
cx q[14],q[5];
rz(pi/4) q[5];
cx q[14],q[5];
cx q[13],q[14];
cx q[16],q[13];
cx q[17],q[16];
cx q[17],q[21];
cx q[16],q[17];
rx(3*pi/4) q[16];
cx q[16],q[17];
cx q[17],q[21];
cx q[7],q[12];
cx q[2],q[7];
cx q[2],q[3];
rx(pi/4) q[2];
cx q[2],q[3];
cx q[2],q[7];
cx q[7],q[12];
cx q[11],q[21];
rx(3*pi/4) q[11];
cx q[11],q[21];
cx q[13],q[12];
rz(pi/2) q[12];
cx q[13],q[12];
rz(3*pi/2) q[3];
rz(3*pi/4) q[16];
rz(pi) q[8];
rz(pi) q[19];
cx q[23],q[16];
cx q[17],q[16];
rz(pi) q[16];
cx q[17],q[16];
cx q[23],q[16];
rx(pi) q[21];
cx q[22],q[17];
rz(3*pi/2) q[17];
cx q[22],q[17];
rz(5*pi/4) q[14];
rz(3*pi/4) q[2];
cx q[8],q[7];
cx q[12],q[7];
rz(5*pi/4) q[7];
cx q[12],q[7];
cx q[8],q[7];
rz(3*pi/2) q[4];
rz(pi/2) q[19];
cx q[6],q[11];
rx(3*pi/2) q[6];
cx q[6],q[11];
cx q[7],q[12];
cx q[1],q[7];
rx(pi/2) q[1];
cx q[1],q[7];
cx q[7],q[12];
cx q[5],q[9];
rx(5*pi/4) q[5];
cx q[5],q[9];
cx q[10],q[20];
rx(3*pi/4) q[10];
cx q[10],q[20];
cx q[3],q[23];
cx q[2],q[3];
rx(pi/4) q[2];
cx q[2],q[3];
cx q[3],q[23];
rx(5*pi/4) q[5];
rx(7*pi/4) q[22];
rx(5*pi/4) q[15];
cx q[16],q[13];
cx q[13],q[6];
rz(pi/4) q[6];
cx q[13],q[6];
cx q[16],q[13];
rz(5*pi/4) q[11];
rz(7*pi/4) q[3];
rz(pi/4) q[12];
cx q[19],q[9];
rz(5*pi/4) q[9];
cx q[19],q[9];
cx q[23],q[16];
cx q[16],q[17];
cx q[17],q[8];
rz(pi/2) q[8];
cx q[17],q[8];
cx q[16],q[17];
cx q[23],q[16];
cx q[17],q[16];
cx q[16],q[13];
cx q[13],q[14];
cx q[14],q[5];
rz(pi/4) q[5];
cx q[14],q[5];
cx q[13],q[14];
cx q[16],q[13];
cx q[17],q[16];
cx q[17],q[21];
cx q[16],q[17];
rx(3*pi/4) q[16];
cx q[16],q[17];
cx q[17],q[21];
cx q[7],q[12];
cx q[2],q[7];
cx q[2],q[3];
rx(pi/4) q[2];
cx q[2],q[3];
cx q[2],q[7];
cx q[7],q[12];
cx q[11],q[21];
rx(3*pi/4) q[11];
cx q[11],q[21];
cx q[13],q[12];
rz(pi/2) q[12];
cx q[13],q[12];
rz(3*pi/2) q[3];
rz(3*pi/4) q[16];
rz(pi) q[8];
rz(pi) q[19];
cx q[23],q[16];
cx q[17],q[16];
rz(pi) q[16];
cx q[17],q[16];
cx q[23],q[16];
rx(pi) q[21];
cx q[22],q[17];
rz(3*pi/2) q[17];
cx q[22],q[17];
rz(5*pi/4) q[14];
rz(3*pi/4) q[2];
cx q[8],q[7];
cx q[12],q[7];
rz(5*pi/4) q[7];
cx q[12],q[7];
cx q[8],q[7];
rz(3*pi/2) q[4];
rz(pi/2) q[19];
cx q[6],q[11];
rx(3*pi/2) q[6];
cx q[6],q[11];
cx q[7],q[12];
cx q[1],q[7];
rx(pi/2) q[1];
cx q[1],q[7];
cx q[7],q[12];
cx q[5],q[9];
rx(5*pi/4) q[5];
cx q[5],q[9];
cx q[10],q[20];
rx(3*pi/4) q[10];
cx q[10],q[20];
cx q[3],q[23];
cx q[2],q[3];
rx(pi/4) q[2];
cx q[2],q[3];
cx q[3],q[23];
rx(5*pi/4) q[5];
rx(7*pi/4) q[22];
rx(5*pi/4) q[15];
cx q[16],q[13];
cx q[13],q[6];
rz(pi/4) q[6];
cx q[13],q[6];
cx q[16],q[13];
rz(5*pi/4) q[11];
rz(7*pi/4) q[3];
rz(pi/4) q[12];
cx q[19],q[9];
rz(5*pi/4) q[9];
cx q[19],q[9];
cx q[23],q[16];
cx q[16],q[17];
cx q[17],q[8];
rz(pi/2) q[8];
cx q[17],q[8];
cx q[16],q[17];
cx q[23],q[16];
cx q[17],q[16];
cx q[16],q[13];
cx q[13],q[14];
cx q[14],q[5];
rz(pi/4) q[5];
cx q[14],q[5];
cx q[13],q[14];
cx q[16],q[13];
cx q[17],q[16];
cx q[17],q[21];
cx q[16],q[17];
rx(3*pi/4) q[16];
cx q[16],q[17];
cx q[17],q[21];
cx q[7],q[12];
cx q[2],q[7];
cx q[2],q[3];
rx(pi/4) q[2];
cx q[2],q[3];
cx q[2],q[7];
cx q[7],q[12];
cx q[11],q[21];
rx(3*pi/4) q[11];
cx q[11],q[21];
cx q[13],q[12];
rz(pi/2) q[12];
cx q[13],q[12];
rz(3*pi/2) q[3];
rz(3*pi/4) q[16];
rz(pi) q[8];
rz(pi) q[19];
cx q[23],q[16];
cx q[17],q[16];
rz(pi) q[16];
cx q[17],q[16];
cx q[23],q[16];
rx(pi) q[21];
cx q[22],q[17];
rz(3*pi/2) q[17];
cx q[22],q[17];
rz(5*pi/4) q[14];
rz(3*pi/4) q[2];
cx q[8],q[7];
cx q[12],q[7];
rz(5*pi/4) q[7];
cx q[12],q[7];
cx q[8],q[7];
rz(3*pi/2) q[4];
rz(pi/2) q[19];
cx q[6],q[11];
rx(3*pi/2) q[6];
cx q[6],q[11];
cx q[7],q[12];
cx q[1],q[7];
rx(pi/2) q[1];
cx q[1],q[7];
cx q[7],q[12];
cx q[5],q[9];
rx(5*pi/4) q[5];
cx q[5],q[9];
cx q[10],q[20];
rx(3*pi/4) q[10];
cx q[10],q[20];
cx q[3],q[23];
cx q[2],q[3];
rx(pi/4) q[2];
cx q[2],q[3];
cx q[3],q[23];
rx(5*pi/4) q[5];
rx(7*pi/4) q[22];
rx(5*pi/4) q[15];
cx q[16],q[13];
cx q[13],q[6];
rz(pi/4) q[6];
cx q[13],q[6];
cx q[16],q[13];
rz(5*pi/4) q[11];
rz(7*pi/4) q[3];
rz(pi/4) q[12];
cx q[19],q[9];
rz(5*pi/4) q[9];
cx q[19],q[9];
cx q[23],q[16];
cx q[16],q[17];
cx q[17],q[8];
rz(pi/2) q[8];
cx q[17],q[8];
cx q[16],q[17];
cx q[23],q[16];
cx q[17],q[16];
cx q[16],q[13];
cx q[13],q[14];
cx q[14],q[5];
rz(pi/4) q[5];
cx q[14],q[5];
cx q[13],q[14];
cx q[16],q[13];
cx q[17],q[16];
cx q[17],q[21];
cx q[16],q[17];
rx(3*pi/4) q[16];
cx q[16],q[17];
cx q[17],q[21];
cx q[7],q[12];
cx q[2],q[7];
cx q[2],q[3];
rx(pi/4) q[2];
cx q[2],q[3];
cx q[2],q[7];
cx q[7],q[12];
cx q[11],q[21];
rx(3*pi/4) q[11];
cx q[11],q[21];
cx q[13],q[12];
rz(pi/2) q[12];
cx q[13],q[12];
rz(3*pi/2) q[3];
rz(3*pi/4) q[16];
cx q[5],q[14];
cx q[20],q[21];
cx q[12],q[7];
cx q[22],q[23];
cx q[17],q[22];
cx q[16],q[15];
cx q[13],q[6];
cx q[9],q[8];
cx q[14],q[5];
cx q[13],q[14];
cx q[11],q[10];
cx q[22],q[23];
cx q[7],q[6];
cx q[1],q[0];
cx q[3],q[2];
cx q[16],q[17];
cx q[20],q[21];
