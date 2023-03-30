OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
cx q[12],q[17];
cx q[9],q[0];
cx q[19],q[20];
cx q[13],q[16];
cx q[23],q[22];
cx q[3],q[6];
cx q[11],q[18];
cx q[7],q[8];
cx q[19],q[18];
cx q[10],q[11];
cx q[12],q[7];
cx q[8],q[1];
cx q[18],q[17];
cx q[3],q[4];
cx q[14],q[13];
cx q[16],q[15];
cx q[20],q[19];
cx q[7],q[2];
cx q[11],q[8];
cx q[7],q[8];
cx q[8],q[1];
rz(pi) q[1];
cx q[8],q[1];
cx q[7],q[8];
rz(pi) q[5];
rz(pi) q[0];
rz(pi) q[12];
cx q[14],q[13];
rz(pi) q[13];
cx q[14],q[13];
cx q[15],q[16];
cx q[16],q[13];
rz(pi) q[13];
cx q[16],q[13];
cx q[15],q[16];
cx q[19],q[18];
cx q[18],q[11];
rz(pi) q[11];
cx q[18],q[11];
cx q[19],q[18];
cx q[17],q[12];
rz(pi) q[12];
cx q[17],q[12];
cx q[20],q[19];
rz(pi) q[19];
cx q[20],q[19];
rz(pi) q[24];
cx q[11],q[18];
cx q[10],q[11];
rx(pi) q[10];
cx q[10],q[11];
cx q[11],q[18];
cx q[19],q[20];
cx q[18],q[19];
rx(pi) q[18];
cx q[18],q[19];
cx q[19],q[20];
rx(pi) q[23];
rx(pi) q[24];
cx q[22],q[17];
cx q[17],q[12];
rz(pi/4) q[12];
cx q[17],q[12];
cx q[22],q[17];
cx q[14],q[13];
cx q[13],q[9];
rz(3*pi/2) q[9];
cx q[13],q[9];
cx q[14],q[13];
rz(3*pi/4) q[6];
cx q[16],q[13];
cx q[14],q[13];
rz(5*pi/4) q[13];
cx q[14],q[13];
cx q[16],q[13];
rz(pi/4) q[3];
rz(3*pi/2) q[10];
cx q[15],q[21];
rx(5*pi/4) q[15];
cx q[15],q[21];
cx q[8],q[21];
cx q[7],q[8];
cx q[2],q[7];
rx(pi/4) q[2];
cx q[2],q[7];
cx q[7],q[8];
cx q[8],q[21];
cx q[11],q[19];
cx q[8],q[11];
rx(7*pi/4) q[8];
cx q[8],q[11];
cx q[11],q[19];
cx q[1],q[8];
rx(3*pi/2) q[1];
cx q[1],q[8];
cx q[19],q[20];
cx q[18],q[19];
cx q[17],q[24];
cx q[17],q[18];
cx q[8],q[9];
cx q[12],q[17];
cx q[7],q[8];
cx q[7],q[12];
rx(3*pi/2) q[7];
cx q[7],q[12];
cx q[7],q[8];
cx q[12],q[17];
cx q[8],q[9];
cx q[17],q[18];
cx q[17],q[24];
cx q[18],q[19];
cx q[19],q[20];
rx(3*pi/2) q[15];
cx q[5],q[17];
rx(5*pi/4) q[5];
cx q[5],q[17];
rx(3*pi/2) q[0];
cx q[13],q[14];
cx q[16],q[13];
cx q[18],q[16];
cx q[19],q[20];
cx q[19],q[18];
cx q[9],q[19];
cx q[0],q[9];
rx(pi/4) q[0];
cx q[0],q[9];
cx q[9],q[19];
cx q[19],q[18];
cx q[19],q[20];
cx q[18],q[16];
cx q[16],q[13];
cx q[13],q[14];
cx q[0],q[9];
rx(pi/2) q[0];
cx q[0],q[9];
cx q[16],q[23];
cx q[13],q[16];
cx q[6],q[13];
cx q[0],q[6];
rx(pi/4) q[0];
cx q[0],q[6];
cx q[6],q[13];
cx q[13],q[16];
cx q[16],q[23];
rx(3*pi/2) q[6];
cx q[12],q[7];
rz(5*pi/4) q[7];
cx q[12],q[7];
rz(pi/2) q[24];
cx q[20],q[19];
rz(3*pi/4) q[19];
cx q[20],q[19];
cx q[17],q[12];
cx q[12],q[10];
rz(pi/2) q[10];
cx q[12],q[10];
cx q[17],q[12];
rz(3*pi/2) q[12];
rz(pi/2) q[0];
rz(pi/2) q[14];
cx q[23],q[11];
cx q[11],q[10];
rz(3*pi/2) q[10];
cx q[11],q[10];
cx q[23],q[11];
cx q[2],q[7];
cx q[8],q[21];
cx q[1],q[2];
cx q[1],q[8];
rx(3*pi/4) q[1];
cx q[1],q[8];
cx q[1],q[2];
cx q[8],q[21];
cx q[2],q[7];
cx q[18],q[19];
cx q[15],q[24];
cx q[16],q[18];
cx q[16],q[15];
cx q[2],q[16];
rx(pi/4) q[2];
cx q[2],q[16];
cx q[16],q[15];
cx q[16],q[18];
cx q[15],q[24];
cx q[18],q[19];
cx q[17],q[22];
cx q[12],q[17];
cx q[7],q[8];
cx q[7],q[12];
cx q[6],q[7];
rx(pi/4) q[6];
cx q[6],q[7];
cx q[7],q[12];
cx q[7],q[8];
cx q[12],q[17];
cx q[17],q[22];
cx q[15],q[16];
cx q[16],q[13];
cx q[13],q[12];
cx q[10],q[12];
cx q[12],q[7];
rz(7*pi/4) q[7];
cx q[12],q[7];
cx q[10],q[12];
cx q[13],q[12];
cx q[16],q[13];
cx q[15],q[16];
cx q[14],q[7];
cx q[9],q[8];
cx q[7],q[8];
cx q[8],q[1];
rz(5*pi/4) q[1];
cx q[8],q[1];
cx q[7],q[8];
cx q[9],q[8];
cx q[14],q[7];
cx q[24],q[5];
rz(3*pi/4) q[5];
cx q[24],q[5];
rx(7*pi/4) q[15];
cx q[9],q[18];
rx(7*pi/4) q[9];
cx q[9],q[18];
cx q[18],q[22];
cx q[19],q[18];
cx q[9],q[19];
rx(3*pi/4) q[9];
cx q[9],q[19];
cx q[19],q[18];
cx q[18],q[22];
cx q[15],q[17];
rx(pi/2) q[15];
cx q[15],q[17];
cx q[20],q[19];
cx q[19],q[16];
rz(7*pi/4) q[16];
cx q[19],q[16];
cx q[20],q[19];
cx q[12],q[17];
cx q[17],q[18];
cx q[18],q[19];
cx q[19],q[10];
cx q[10],q[9];
rz(3*pi/2) q[9];
cx q[10],q[9];
cx q[19],q[10];
cx q[18],q[19];
cx q[17],q[18];
cx q[12],q[17];
cx q[13],q[16];
cx q[13],q[14];
rx(3*pi/4) q[13];
cx q[13],q[14];
cx q[13],q[16];
cx q[16],q[13];
cx q[18],q[16];
cx q[18],q[21];
cx q[11],q[18];
cx q[10],q[11];
rx(pi/2) q[10];
cx q[10],q[11];
cx q[11],q[18];
cx q[18],q[21];
cx q[18],q[16];
cx q[16],q[13];
rx(3*pi/2) q[17];
cx q[17],q[12];
cx q[4],q[3];
cx q[12],q[3];
rz(3*pi/4) q[3];
cx q[12],q[3];
cx q[4],q[3];
cx q[17],q[12];
cx q[19],q[10];
cx q[18],q[11];
cx q[10],q[11];
cx q[11],q[12];
cx q[12],q[7];
cx q[5],q[7];
cx q[7],q[2];
rz(pi/2) q[2];
cx q[7],q[2];
cx q[5],q[7];
cx q[12],q[7];
cx q[11],q[12];
cx q[10],q[11];
cx q[18],q[11];
cx q[19],q[10];
cx q[7],q[8];
cx q[8],q[1];
rz(pi) q[1];
cx q[8],q[1];
cx q[7],q[8];
rz(pi) q[5];
rz(pi) q[0];
rz(pi) q[12];
cx q[14],q[13];
rz(pi) q[13];
cx q[14],q[13];
cx q[15],q[16];
cx q[16],q[13];
rz(pi) q[13];
cx q[16],q[13];
cx q[15],q[16];
cx q[19],q[18];
cx q[18],q[11];
rz(pi) q[11];
cx q[18],q[11];
cx q[19],q[18];
cx q[17],q[12];
rz(pi) q[12];
cx q[17],q[12];
cx q[20],q[19];
rz(pi) q[19];
cx q[20],q[19];
rz(pi) q[24];
cx q[11],q[18];
cx q[10],q[11];
rx(pi) q[10];
cx q[10],q[11];
cx q[11],q[18];
cx q[19],q[20];
cx q[18],q[19];
rx(pi) q[18];
cx q[18],q[19];
cx q[19],q[20];
rx(pi) q[23];
rx(pi) q[24];
cx q[22],q[17];
cx q[17],q[12];
rz(pi/4) q[12];
cx q[17],q[12];
cx q[22],q[17];
cx q[14],q[13];
cx q[13],q[9];
rz(3*pi/2) q[9];
cx q[13],q[9];
cx q[14],q[13];
rz(3*pi/4) q[6];
cx q[16],q[13];
cx q[14],q[13];
rz(5*pi/4) q[13];
cx q[14],q[13];
cx q[16],q[13];
rz(pi/4) q[3];
rz(3*pi/2) q[10];
cx q[15],q[21];
rx(5*pi/4) q[15];
cx q[15],q[21];
cx q[8],q[21];
cx q[7],q[8];
cx q[2],q[7];
rx(pi/4) q[2];
cx q[2],q[7];
cx q[7],q[8];
cx q[8],q[21];
cx q[11],q[19];
cx q[8],q[11];
rx(7*pi/4) q[8];
cx q[8],q[11];
cx q[11],q[19];
cx q[1],q[8];
rx(3*pi/2) q[1];
cx q[1],q[8];
cx q[19],q[20];
cx q[18],q[19];
cx q[17],q[24];
cx q[17],q[18];
cx q[8],q[9];
cx q[12],q[17];
cx q[7],q[8];
cx q[7],q[12];
rx(3*pi/2) q[7];
cx q[7],q[12];
cx q[7],q[8];
cx q[12],q[17];
cx q[8],q[9];
cx q[17],q[18];
cx q[17],q[24];
cx q[18],q[19];
cx q[19],q[20];
rx(3*pi/2) q[15];
cx q[5],q[17];
rx(5*pi/4) q[5];
cx q[5],q[17];
rx(3*pi/2) q[0];
cx q[13],q[14];
cx q[16],q[13];
cx q[18],q[16];
cx q[19],q[20];
cx q[19],q[18];
cx q[9],q[19];
cx q[0],q[9];
rx(pi/4) q[0];
cx q[0],q[9];
cx q[9],q[19];
cx q[19],q[18];
cx q[19],q[20];
cx q[18],q[16];
cx q[16],q[13];
cx q[13],q[14];
cx q[0],q[9];
rx(pi/2) q[0];
cx q[0],q[9];
cx q[16],q[23];
cx q[13],q[16];
cx q[6],q[13];
cx q[0],q[6];
rx(pi/4) q[0];
cx q[0],q[6];
cx q[6],q[13];
cx q[13],q[16];
cx q[16],q[23];
rx(3*pi/2) q[6];
cx q[12],q[7];
rz(5*pi/4) q[7];
cx q[12],q[7];
rz(pi/2) q[24];
cx q[20],q[19];
rz(3*pi/4) q[19];
cx q[20],q[19];
cx q[17],q[12];
cx q[12],q[10];
rz(pi/2) q[10];
cx q[12],q[10];
cx q[17],q[12];
rz(3*pi/2) q[12];
rz(pi/2) q[0];
rz(pi/2) q[14];
cx q[23],q[11];
cx q[11],q[10];
rz(3*pi/2) q[10];
cx q[11],q[10];
cx q[23],q[11];
cx q[2],q[7];
cx q[8],q[21];
cx q[1],q[2];
cx q[1],q[8];
rx(3*pi/4) q[1];
cx q[1],q[8];
cx q[1],q[2];
cx q[8],q[21];
cx q[2],q[7];
cx q[18],q[19];
cx q[15],q[24];
cx q[16],q[18];
cx q[16],q[15];
cx q[2],q[16];
rx(pi/4) q[2];
cx q[2],q[16];
cx q[16],q[15];
cx q[16],q[18];
cx q[15],q[24];
cx q[18],q[19];
cx q[17],q[22];
cx q[12],q[17];
cx q[7],q[8];
cx q[7],q[12];
cx q[6],q[7];
rx(pi/4) q[6];
cx q[6],q[7];
cx q[7],q[12];
cx q[7],q[8];
cx q[12],q[17];
cx q[17],q[22];
cx q[15],q[16];
cx q[16],q[13];
cx q[13],q[12];
cx q[10],q[12];
cx q[12],q[7];
rz(7*pi/4) q[7];
cx q[12],q[7];
cx q[10],q[12];
cx q[13],q[12];
cx q[16],q[13];
cx q[15],q[16];
cx q[14],q[7];
cx q[9],q[8];
cx q[7],q[8];
cx q[8],q[1];
rz(5*pi/4) q[1];
cx q[8],q[1];
cx q[7],q[8];
cx q[9],q[8];
cx q[14],q[7];
cx q[24],q[5];
rz(3*pi/4) q[5];
cx q[24],q[5];
rx(7*pi/4) q[15];
cx q[9],q[18];
rx(7*pi/4) q[9];
cx q[9],q[18];
cx q[18],q[22];
cx q[19],q[18];
cx q[9],q[19];
rx(3*pi/4) q[9];
cx q[9],q[19];
cx q[19],q[18];
cx q[18],q[22];
cx q[15],q[17];
rx(pi/2) q[15];
cx q[15],q[17];
cx q[20],q[19];
cx q[19],q[16];
rz(7*pi/4) q[16];
cx q[19],q[16];
cx q[20],q[19];
cx q[12],q[17];
cx q[17],q[18];
cx q[18],q[19];
cx q[19],q[10];
cx q[10],q[9];
rz(3*pi/2) q[9];
cx q[10],q[9];
cx q[19],q[10];
cx q[18],q[19];
cx q[17],q[18];
cx q[12],q[17];
cx q[13],q[16];
cx q[13],q[14];
rx(3*pi/4) q[13];
cx q[13],q[14];
cx q[13],q[16];
cx q[16],q[13];
cx q[18],q[16];
cx q[18],q[21];
cx q[11],q[18];
cx q[10],q[11];
rx(pi/2) q[10];
cx q[10],q[11];
cx q[11],q[18];
cx q[18],q[21];
cx q[18],q[16];
cx q[16],q[13];
rx(3*pi/2) q[17];
cx q[17],q[12];
cx q[4],q[3];
cx q[12],q[3];
rz(3*pi/4) q[3];
cx q[12],q[3];
cx q[4],q[3];
cx q[17],q[12];
cx q[19],q[10];
cx q[18],q[11];
cx q[10],q[11];
cx q[11],q[12];
cx q[12],q[7];
cx q[5],q[7];
cx q[7],q[2];
rz(pi/2) q[2];
cx q[7],q[2];
cx q[5],q[7];
cx q[12],q[7];
cx q[11],q[12];
cx q[10],q[11];
cx q[18],q[11];
cx q[19],q[10];
cx q[7],q[8];
cx q[8],q[1];
rz(pi) q[1];
cx q[8],q[1];
cx q[7],q[8];
rz(pi) q[5];
rz(pi) q[0];
rz(pi) q[12];
cx q[14],q[13];
rz(pi) q[13];
cx q[14],q[13];
cx q[15],q[16];
cx q[16],q[13];
rz(pi) q[13];
cx q[16],q[13];
cx q[15],q[16];
cx q[19],q[18];
cx q[18],q[11];
rz(pi) q[11];
cx q[18],q[11];
cx q[19],q[18];
cx q[17],q[12];
rz(pi) q[12];
cx q[17],q[12];
cx q[20],q[19];
rz(pi) q[19];
cx q[20],q[19];
rz(pi) q[24];
cx q[11],q[18];
cx q[10],q[11];
rx(pi) q[10];
cx q[10],q[11];
cx q[11],q[18];
cx q[19],q[20];
cx q[18],q[19];
rx(pi) q[18];
cx q[18],q[19];
cx q[19],q[20];
rx(pi) q[23];
rx(pi) q[24];
cx q[22],q[17];
cx q[17],q[12];
rz(pi/4) q[12];
cx q[17],q[12];
cx q[22],q[17];
cx q[14],q[13];
cx q[13],q[9];
rz(3*pi/2) q[9];
cx q[13],q[9];
cx q[14],q[13];
rz(3*pi/4) q[6];
cx q[16],q[13];
cx q[14],q[13];
rz(5*pi/4) q[13];
cx q[14],q[13];
cx q[16],q[13];
rz(pi/4) q[3];
rz(3*pi/2) q[10];
cx q[15],q[21];
rx(5*pi/4) q[15];
cx q[15],q[21];
cx q[8],q[21];
cx q[7],q[8];
cx q[2],q[7];
rx(pi/4) q[2];
cx q[2],q[7];
cx q[7],q[8];
cx q[8],q[21];
cx q[11],q[19];
cx q[8],q[11];
rx(7*pi/4) q[8];
cx q[8],q[11];
cx q[11],q[19];
cx q[1],q[8];
rx(3*pi/2) q[1];
cx q[1],q[8];
cx q[19],q[20];
cx q[18],q[19];
cx q[17],q[24];
cx q[17],q[18];
cx q[8],q[9];
cx q[12],q[17];
cx q[7],q[8];
cx q[7],q[12];
rx(3*pi/2) q[7];
cx q[7],q[12];
cx q[7],q[8];
cx q[12],q[17];
cx q[8],q[9];
cx q[17],q[18];
cx q[17],q[24];
cx q[18],q[19];
cx q[19],q[20];
rx(3*pi/2) q[15];
cx q[5],q[17];
rx(5*pi/4) q[5];
cx q[5],q[17];
rx(3*pi/2) q[0];
cx q[13],q[14];
cx q[16],q[13];
cx q[18],q[16];
cx q[19],q[20];
cx q[19],q[18];
cx q[9],q[19];
cx q[0],q[9];
rx(pi/4) q[0];
cx q[0],q[9];
cx q[9],q[19];
cx q[19],q[18];
cx q[19],q[20];
cx q[18],q[16];
cx q[16],q[13];
cx q[13],q[14];
cx q[0],q[9];
rx(pi/2) q[0];
cx q[0],q[9];
cx q[16],q[23];
cx q[13],q[16];
cx q[6],q[13];
cx q[0],q[6];
rx(pi/4) q[0];
cx q[0],q[6];
cx q[6],q[13];
cx q[13],q[16];
cx q[16],q[23];
rx(3*pi/2) q[6];
cx q[12],q[7];
rz(5*pi/4) q[7];
cx q[12],q[7];
rz(pi/2) q[24];
cx q[20],q[19];
rz(3*pi/4) q[19];
cx q[20],q[19];
cx q[17],q[12];
cx q[12],q[10];
rz(pi/2) q[10];
cx q[12],q[10];
cx q[17],q[12];
rz(3*pi/2) q[12];
rz(pi/2) q[0];
rz(pi/2) q[14];
cx q[23],q[11];
cx q[11],q[10];
rz(3*pi/2) q[10];
cx q[11],q[10];
cx q[23],q[11];
cx q[2],q[7];
cx q[8],q[21];
cx q[1],q[2];
cx q[1],q[8];
rx(3*pi/4) q[1];
cx q[1],q[8];
cx q[1],q[2];
cx q[8],q[21];
cx q[2],q[7];
cx q[18],q[19];
cx q[15],q[24];
cx q[16],q[18];
cx q[16],q[15];
cx q[2],q[16];
rx(pi/4) q[2];
cx q[2],q[16];
cx q[16],q[15];
cx q[16],q[18];
cx q[15],q[24];
cx q[18],q[19];
cx q[17],q[22];
cx q[12],q[17];
cx q[7],q[8];
cx q[7],q[12];
cx q[6],q[7];
rx(pi/4) q[6];
cx q[6],q[7];
cx q[7],q[12];
cx q[7],q[8];
cx q[12],q[17];
cx q[17],q[22];
cx q[15],q[16];
cx q[16],q[13];
cx q[13],q[12];
cx q[10],q[12];
cx q[12],q[7];
rz(7*pi/4) q[7];
cx q[12],q[7];
cx q[10],q[12];
cx q[13],q[12];
cx q[16],q[13];
cx q[15],q[16];
cx q[14],q[7];
cx q[9],q[8];
cx q[7],q[8];
cx q[8],q[1];
rz(5*pi/4) q[1];
cx q[8],q[1];
cx q[7],q[8];
cx q[9],q[8];
cx q[14],q[7];
cx q[24],q[5];
rz(3*pi/4) q[5];
cx q[24],q[5];
rx(7*pi/4) q[15];
cx q[9],q[18];
rx(7*pi/4) q[9];
cx q[9],q[18];
cx q[18],q[22];
cx q[19],q[18];
cx q[9],q[19];
rx(3*pi/4) q[9];
cx q[9],q[19];
cx q[19],q[18];
cx q[18],q[22];
cx q[15],q[17];
rx(pi/2) q[15];
cx q[15],q[17];
cx q[20],q[19];
cx q[19],q[16];
rz(7*pi/4) q[16];
cx q[19],q[16];
cx q[20],q[19];
cx q[12],q[17];
cx q[17],q[18];
cx q[18],q[19];
cx q[19],q[10];
cx q[10],q[9];
rz(3*pi/2) q[9];
cx q[10],q[9];
cx q[19],q[10];
cx q[18],q[19];
cx q[17],q[18];
cx q[12],q[17];
cx q[13],q[16];
cx q[13],q[14];
rx(3*pi/4) q[13];
cx q[13],q[14];
cx q[13],q[16];
cx q[16],q[13];
cx q[18],q[16];
cx q[18],q[21];
cx q[11],q[18];
cx q[10],q[11];
rx(pi/2) q[10];
cx q[10],q[11];
cx q[11],q[18];
cx q[18],q[21];
cx q[18],q[16];
cx q[16],q[13];
rx(3*pi/2) q[17];
cx q[17],q[12];
cx q[4],q[3];
cx q[12],q[3];
rz(3*pi/4) q[3];
cx q[12],q[3];
cx q[4],q[3];
cx q[17],q[12];
cx q[19],q[10];
cx q[18],q[11];
cx q[10],q[11];
cx q[11],q[12];
cx q[12],q[7];
cx q[5],q[7];
cx q[7],q[2];
rz(pi/2) q[2];
cx q[7],q[2];
cx q[5],q[7];
cx q[12],q[7];
cx q[11],q[12];
cx q[10],q[11];
cx q[18],q[11];
cx q[19],q[10];
cx q[7],q[8];
cx q[8],q[1];
rz(pi) q[1];
cx q[8],q[1];
cx q[7],q[8];
rz(pi) q[5];
rz(pi) q[0];
rz(pi) q[12];
cx q[14],q[13];
rz(pi) q[13];
cx q[14],q[13];
cx q[15],q[16];
cx q[16],q[13];
rz(pi) q[13];
cx q[16],q[13];
cx q[15],q[16];
cx q[19],q[18];
cx q[18],q[11];
rz(pi) q[11];
cx q[18],q[11];
cx q[19],q[18];
cx q[17],q[12];
rz(pi) q[12];
cx q[17],q[12];
cx q[20],q[19];
rz(pi) q[19];
cx q[20],q[19];
rz(pi) q[24];
cx q[11],q[18];
cx q[10],q[11];
rx(pi) q[10];
cx q[10],q[11];
cx q[11],q[18];
cx q[19],q[20];
cx q[18],q[19];
rx(pi) q[18];
cx q[18],q[19];
cx q[19],q[20];
rx(pi) q[23];
rx(pi) q[24];
cx q[22],q[17];
cx q[17],q[12];
rz(pi/4) q[12];
cx q[17],q[12];
cx q[22],q[17];
cx q[14],q[13];
cx q[13],q[9];
rz(3*pi/2) q[9];
cx q[13],q[9];
cx q[14],q[13];
rz(3*pi/4) q[6];
cx q[16],q[13];
cx q[14],q[13];
rz(5*pi/4) q[13];
cx q[14],q[13];
cx q[16],q[13];
rz(pi/4) q[3];
rz(3*pi/2) q[10];
cx q[15],q[21];
rx(5*pi/4) q[15];
cx q[15],q[21];
cx q[8],q[21];
cx q[7],q[8];
cx q[2],q[7];
rx(pi/4) q[2];
cx q[2],q[7];
cx q[7],q[8];
cx q[8],q[21];
cx q[11],q[19];
cx q[8],q[11];
rx(7*pi/4) q[8];
cx q[8],q[11];
cx q[11],q[19];
cx q[1],q[8];
rx(3*pi/2) q[1];
cx q[1],q[8];
cx q[19],q[20];
cx q[18],q[19];
cx q[17],q[24];
cx q[17],q[18];
cx q[8],q[9];
cx q[12],q[17];
cx q[7],q[8];
cx q[7],q[12];
rx(3*pi/2) q[7];
cx q[7],q[12];
cx q[7],q[8];
cx q[12],q[17];
cx q[8],q[9];
cx q[17],q[18];
cx q[17],q[24];
cx q[18],q[19];
cx q[19],q[20];
rx(3*pi/2) q[15];
cx q[5],q[17];
rx(5*pi/4) q[5];
cx q[5],q[17];
rx(3*pi/2) q[0];
cx q[13],q[14];
cx q[16],q[13];
cx q[18],q[16];
cx q[19],q[20];
cx q[19],q[18];
cx q[9],q[19];
cx q[0],q[9];
rx(pi/4) q[0];
cx q[0],q[9];
cx q[9],q[19];
cx q[19],q[18];
cx q[19],q[20];
cx q[18],q[16];
cx q[16],q[13];
cx q[13],q[14];
cx q[0],q[9];
rx(pi/2) q[0];
cx q[0],q[9];
cx q[16],q[23];
cx q[13],q[16];
cx q[6],q[13];
cx q[0],q[6];
rx(pi/4) q[0];
cx q[0],q[6];
cx q[6],q[13];
cx q[13],q[16];
cx q[16],q[23];
rx(3*pi/2) q[6];
cx q[12],q[7];
rz(5*pi/4) q[7];
cx q[12],q[7];
rz(pi/2) q[24];
cx q[20],q[19];
rz(3*pi/4) q[19];
cx q[20],q[19];
cx q[17],q[12];
cx q[12],q[10];
rz(pi/2) q[10];
cx q[12],q[10];
cx q[17],q[12];
rz(3*pi/2) q[12];
rz(pi/2) q[0];
rz(pi/2) q[14];
cx q[23],q[11];
cx q[11],q[10];
rz(3*pi/2) q[10];
cx q[11],q[10];
cx q[23],q[11];
cx q[2],q[7];
cx q[8],q[21];
cx q[1],q[2];
cx q[1],q[8];
rx(3*pi/4) q[1];
cx q[1],q[8];
cx q[1],q[2];
cx q[8],q[21];
cx q[2],q[7];
cx q[18],q[19];
cx q[15],q[24];
cx q[16],q[18];
cx q[16],q[15];
cx q[2],q[16];
rx(pi/4) q[2];
cx q[2],q[16];
cx q[16],q[15];
cx q[16],q[18];
cx q[15],q[24];
cx q[18],q[19];
cx q[17],q[22];
cx q[12],q[17];
cx q[7],q[8];
cx q[7],q[12];
cx q[6],q[7];
rx(pi/4) q[6];
cx q[6],q[7];
cx q[7],q[12];
cx q[7],q[8];
cx q[12],q[17];
cx q[17],q[22];
cx q[15],q[16];
cx q[16],q[13];
cx q[13],q[12];
cx q[10],q[12];
cx q[12],q[7];
rz(7*pi/4) q[7];
cx q[12],q[7];
cx q[10],q[12];
cx q[13],q[12];
cx q[16],q[13];
cx q[15],q[16];
cx q[14],q[7];
cx q[9],q[8];
cx q[7],q[8];
cx q[8],q[1];
rz(5*pi/4) q[1];
cx q[8],q[1];
cx q[7],q[8];
cx q[9],q[8];
cx q[14],q[7];
cx q[24],q[5];
rz(3*pi/4) q[5];
cx q[24],q[5];
rx(7*pi/4) q[15];
cx q[9],q[18];
rx(7*pi/4) q[9];
cx q[9],q[18];
cx q[18],q[22];
cx q[19],q[18];
cx q[9],q[19];
rx(3*pi/4) q[9];
cx q[9],q[19];
cx q[19],q[18];
cx q[18],q[22];
cx q[15],q[17];
rx(pi/2) q[15];
cx q[15],q[17];
cx q[20],q[19];
cx q[19],q[16];
rz(7*pi/4) q[16];
cx q[19],q[16];
cx q[20],q[19];
cx q[12],q[17];
cx q[17],q[18];
cx q[18],q[19];
cx q[19],q[10];
cx q[10],q[9];
rz(3*pi/2) q[9];
cx q[10],q[9];
cx q[19],q[10];
cx q[18],q[19];
cx q[17],q[18];
cx q[12],q[17];
cx q[13],q[16];
cx q[13],q[14];
rx(3*pi/4) q[13];
cx q[13],q[14];
cx q[13],q[16];
cx q[16],q[13];
cx q[18],q[16];
cx q[18],q[21];
cx q[11],q[18];
cx q[10],q[11];
rx(pi/2) q[10];
cx q[10],q[11];
cx q[11],q[18];
cx q[18],q[21];
cx q[18],q[16];
cx q[16],q[13];
rx(3*pi/2) q[17];
cx q[17],q[12];
cx q[4],q[3];
cx q[12],q[3];
rz(3*pi/4) q[3];
cx q[12],q[3];
cx q[4],q[3];
cx q[17],q[12];
cx q[19],q[10];
cx q[18],q[11];
cx q[10],q[11];
cx q[11],q[12];
cx q[12],q[7];
cx q[5],q[7];
cx q[7],q[2];
rz(pi/2) q[2];
cx q[7],q[2];
cx q[5],q[7];
cx q[12],q[7];
cx q[11],q[12];
cx q[10],q[11];
cx q[18],q[11];
cx q[19],q[10];
cx q[7],q[8];
cx q[8],q[1];
rz(pi) q[1];
cx q[8],q[1];
cx q[7],q[8];
rz(pi) q[5];
rz(pi) q[0];
rz(pi) q[12];
cx q[14],q[13];
rz(pi) q[13];
cx q[14],q[13];
cx q[15],q[16];
cx q[16],q[13];
rz(pi) q[13];
cx q[16],q[13];
cx q[15],q[16];
cx q[19],q[18];
cx q[18],q[11];
rz(pi) q[11];
cx q[18],q[11];
cx q[19],q[18];
cx q[17],q[12];
rz(pi) q[12];
cx q[17],q[12];
cx q[20],q[19];
rz(pi) q[19];
cx q[20],q[19];
rz(pi) q[24];
cx q[11],q[18];
cx q[10],q[11];
rx(pi) q[10];
cx q[10],q[11];
cx q[11],q[18];
cx q[19],q[20];
cx q[18],q[19];
rx(pi) q[18];
cx q[18],q[19];
cx q[19],q[20];
rx(pi) q[23];
rx(pi) q[24];
cx q[22],q[17];
cx q[17],q[12];
rz(pi/4) q[12];
cx q[17],q[12];
cx q[22],q[17];
cx q[14],q[13];
cx q[13],q[9];
rz(3*pi/2) q[9];
cx q[13],q[9];
cx q[14],q[13];
rz(3*pi/4) q[6];
cx q[16],q[13];
cx q[14],q[13];
rz(5*pi/4) q[13];
cx q[14],q[13];
cx q[16],q[13];
rz(pi/4) q[3];
rz(3*pi/2) q[10];
cx q[15],q[21];
rx(5*pi/4) q[15];
cx q[15],q[21];
cx q[8],q[21];
cx q[7],q[8];
cx q[2],q[7];
rx(pi/4) q[2];
cx q[2],q[7];
cx q[7],q[8];
cx q[8],q[21];
cx q[11],q[19];
cx q[8],q[11];
rx(7*pi/4) q[8];
cx q[8],q[11];
cx q[11],q[19];
cx q[1],q[8];
rx(3*pi/2) q[1];
cx q[1],q[8];
cx q[19],q[20];
cx q[18],q[19];
cx q[17],q[24];
cx q[17],q[18];
cx q[8],q[9];
cx q[12],q[17];
cx q[7],q[8];
cx q[7],q[12];
rx(3*pi/2) q[7];
cx q[7],q[12];
cx q[7],q[8];
cx q[12],q[17];
cx q[8],q[9];
cx q[17],q[18];
cx q[17],q[24];
cx q[18],q[19];
cx q[19],q[20];
rx(3*pi/2) q[15];
cx q[5],q[17];
rx(5*pi/4) q[5];
cx q[5],q[17];
rx(3*pi/2) q[0];
cx q[13],q[14];
cx q[16],q[13];
cx q[18],q[16];
cx q[19],q[20];
cx q[19],q[18];
cx q[9],q[19];
cx q[0],q[9];
rx(pi/4) q[0];
cx q[0],q[9];
cx q[9],q[19];
cx q[19],q[18];
cx q[19],q[20];
cx q[18],q[16];
cx q[16],q[13];
cx q[13],q[14];
cx q[0],q[9];
rx(pi/2) q[0];
cx q[0],q[9];
cx q[16],q[23];
cx q[13],q[16];
cx q[6],q[13];
cx q[0],q[6];
rx(pi/4) q[0];
cx q[0],q[6];
cx q[6],q[13];
cx q[13],q[16];
cx q[16],q[23];
rx(3*pi/2) q[6];
cx q[12],q[7];
rz(5*pi/4) q[7];
cx q[12],q[7];
rz(pi/2) q[24];
cx q[20],q[19];
rz(3*pi/4) q[19];
cx q[20],q[19];
cx q[17],q[12];
cx q[12],q[10];
rz(pi/2) q[10];
cx q[12],q[10];
cx q[17],q[12];
rz(3*pi/2) q[12];
rz(pi/2) q[0];
rz(pi/2) q[14];
cx q[23],q[11];
cx q[11],q[10];
rz(3*pi/2) q[10];
cx q[11],q[10];
cx q[23],q[11];
cx q[2],q[7];
cx q[8],q[21];
cx q[1],q[2];
cx q[1],q[8];
rx(3*pi/4) q[1];
cx q[1],q[8];
cx q[1],q[2];
cx q[8],q[21];
cx q[2],q[7];
cx q[18],q[19];
cx q[15],q[24];
cx q[16],q[18];
cx q[16],q[15];
cx q[2],q[16];
rx(pi/4) q[2];
cx q[2],q[16];
cx q[16],q[15];
cx q[16],q[18];
cx q[15],q[24];
cx q[18],q[19];
cx q[17],q[22];
cx q[12],q[17];
cx q[7],q[8];
cx q[7],q[12];
cx q[6],q[7];
rx(pi/4) q[6];
cx q[6],q[7];
cx q[7],q[12];
cx q[7],q[8];
cx q[12],q[17];
cx q[17],q[22];
cx q[15],q[16];
cx q[16],q[13];
cx q[13],q[12];
cx q[10],q[12];
cx q[12],q[7];
rz(7*pi/4) q[7];
cx q[12],q[7];
cx q[10],q[12];
cx q[13],q[12];
cx q[16],q[13];
cx q[15],q[16];
cx q[14],q[7];
cx q[9],q[8];
cx q[7],q[8];
cx q[8],q[1];
rz(5*pi/4) q[1];
cx q[8],q[1];
cx q[7],q[8];
cx q[9],q[8];
cx q[14],q[7];
cx q[24],q[5];
rz(3*pi/4) q[5];
cx q[24],q[5];
rx(7*pi/4) q[15];
cx q[9],q[18];
rx(7*pi/4) q[9];
cx q[9],q[18];
cx q[18],q[22];
cx q[19],q[18];
cx q[9],q[19];
rx(3*pi/4) q[9];
cx q[9],q[19];
cx q[19],q[18];
cx q[18],q[22];
cx q[15],q[17];
rx(pi/2) q[15];
cx q[15],q[17];
cx q[20],q[19];
cx q[19],q[16];
rz(7*pi/4) q[16];
cx q[19],q[16];
cx q[20],q[19];
cx q[12],q[17];
cx q[17],q[18];
cx q[18],q[19];
cx q[19],q[10];
cx q[10],q[9];
rz(3*pi/2) q[9];
cx q[10],q[9];
cx q[19],q[10];
cx q[18],q[19];
cx q[17],q[18];
cx q[12],q[17];
cx q[13],q[16];
cx q[13],q[14];
rx(3*pi/4) q[13];
cx q[13],q[14];
cx q[13],q[16];
cx q[16],q[13];
cx q[18],q[16];
cx q[18],q[21];
cx q[11],q[18];
cx q[10],q[11];
rx(pi/2) q[10];
cx q[10],q[11];
cx q[11],q[18];
cx q[18],q[21];
cx q[18],q[16];
cx q[16],q[13];
rx(3*pi/2) q[17];
cx q[17],q[12];
cx q[4],q[3];
cx q[12],q[3];
rz(3*pi/4) q[3];
cx q[12],q[3];
cx q[4],q[3];
cx q[17],q[12];
cx q[19],q[10];
cx q[18],q[11];
cx q[10],q[11];
cx q[11],q[12];
cx q[12],q[7];
cx q[5],q[7];
cx q[7],q[2];
rz(pi/2) q[2];
cx q[7],q[2];
cx q[5],q[7];
cx q[12],q[7];
cx q[11],q[12];
cx q[10],q[11];
cx q[18],q[11];
cx q[19],q[10];
cx q[18],q[17];
cx q[3],q[4];
cx q[14],q[13];
cx q[16],q[15];
cx q[20],q[19];
cx q[7],q[2];
cx q[11],q[8];
cx q[19],q[18];
cx q[10],q[11];
cx q[12],q[7];
cx q[8],q[1];
cx q[12],q[17];
cx q[9],q[0];
cx q[19],q[20];
cx q[13],q[16];
cx q[23],q[22];
cx q[3],q[6];
cx q[11],q[18];
cx q[7],q[8];
