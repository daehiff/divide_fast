OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
cx q[2],q[1];
cx q[3],q[1];
cx q[4],q[1];
cx q[5],q[1];
cx q[8],q[5];
cx q[8],q[4];
cx q[8],q[3];
cx q[8],q[2];
cx q[9],q[5];
cx q[9],q[2];
cx q[9],q[1];
cx q[9],q[0];
cx q[10],q[4];
cx q[11],q[5];
cx q[11],q[3];
cx q[12],q[5];
cx q[13],q[9];
cx q[14],q[1];
cx q[15],q[12];
cx q[15],q[9];
cx q[15],q[5];
cx q[15],q[3];
cx q[15],q[1];
cx q[17],q[16];
cx q[17],q[14];
cx q[17],q[2];
cx q[17],q[1];
cx q[17],q[0];
cx q[18],q[16];
cx q[18],q[7];
cx q[19],q[7];
cx q[21],q[17];
cx q[21],q[5];
cx q[21],q[3];
cx q[22],q[16];
cx q[22],q[12];
cx q[22],q[2];
cx q[22],q[1];
cx q[22],q[0];
cx q[23],q[17];
cx q[23],q[16];
cx q[23],q[14];
cx q[23],q[12];
cx q[23],q[3];
cx q[23],q[2];
cx q[23],q[1];
cx q[24],q[16];
cx q[22],q[23];
cx q[23],q[22];
cx q[21],q[23];
cx q[23],q[21];
cx q[18],q[24];
cx q[18],q[20];
cx q[18],q[19];
cx q[17],q[23];
cx q[17],q[21];
cx q[21],q[17];
cx q[15],q[23];
cx q[15],q[22];
cx q[15],q[21];
cx q[15],q[17];
cx q[15],q[16];
cx q[14],q[23];
cx q[14],q[21];
cx q[14],q[16];
cx q[13],q[23];
cx q[13],q[21];
cx q[13],q[16];
cx q[23],q[13];
cx q[12],q[23];
cx q[12],q[21];
cx q[12],q[16];
cx q[12],q[14];
cx q[12],q[13];
cx q[10],q[23];
cx q[10],q[16];
cx q[9],q[23];
cx q[9],q[21];
cx q[9],q[16];
cx q[9],q[13];
cx q[13],q[9];
cx q[8],q[11];
cx q[5],q[23];
cx q[5],q[21];
cx q[5],q[16];
cx q[5],q[14];
cx q[5],q[13];
cx q[5],q[9];
cx q[9],q[5];
cx q[4],q[23];
cx q[4],q[16];
cx q[23],q[4];
cx q[3],q[23];
cx q[3],q[21];
cx q[3],q[16];
cx q[3],q[14];
cx q[3],q[13];
cx q[3],q[9];
cx q[3],q[4];
cx q[21],q[3];
cx q[2],q[23];
cx q[2],q[16];
cx q[23],q[2];
cx q[1],q[23];
cx q[1],q[22];
cx q[1],q[21];
cx q[1],q[17];
cx q[1],q[16];
cx q[1],q[14];
cx q[1],q[13];
cx q[1],q[12];
cx q[1],q[9];
cx q[1],q[4];
cx q[1],q[2];
cx q[22],q[1];
cx q[0],q[21];
cx q[0],q[14];
cx q[0],q[13];
cx q[0],q[9];
cx q[0],q[4];
cx q[0],q[2];
cx q[14],q[0];
cx q[7],q[5];
cx q[12],q[5];
cx q[17],q[16];
cx q[17],q[5];
cx q[18],q[17];
cx q[18],q[16];
cx q[18],q[5];
cx q[21],q[10];
cx q[21],q[4];
cx q[12],q[17];
cx q[12],q[16];
cx q[7],q[16];
cx q[5],q[16];
cx q[4],q[10];
cx q[11],q[3];
cx q[5],q[3];
cx q[18],q[3];
cx q[7],q[3];
rz(pi) q[3];
cx q[7],q[3];
cx q[18],q[3];
cx q[5],q[3];
cx q[11],q[3];
cx q[11],q[8];
rz(pi) q[8];
cx q[11],q[8];
rz(pi) q[6];
cx q[18],q[17];
rz(pi) q[17];
cx q[18],q[17];
cx q[19],q[18];
rz(pi) q[18];
cx q[19],q[18];
cx q[13],q[10];
rz(pi) q[10];
cx q[13],q[10];
rz(pi) q[8];
cx q[21],q[10];
rz(pi) q[10];
cx q[21],q[10];
cx q[15],q[1];
rz(pi) q[1];
cx q[15],q[1];
cx q[12],q[22];
rx(pi) q[12];
cx q[12],q[22];
cx q[12],q[13];
cx q[7],q[15];
cx q[24],q[18];
rz(pi) q[18];
cx q[24],q[18];
rz(pi) q[18];
rx(pi) q[13];
rx(pi) q[16];
cx q[3],q[11];
rx(pi) q[3];
cx q[3],q[11];
cx q[14],q[22];
rx(pi) q[14];
cx q[14],q[22];
cx q[1],q[23];
rx(pi) q[1];
cx q[1],q[23];
cx q[24],q[0];
cx q[5],q[0];
cx q[13],q[0];
cx q[4],q[0];
cx q[23],q[0];
rz(pi/4) q[0];
cx q[23],q[0];
cx q[4],q[0];
cx q[13],q[0];
cx q[5],q[0];
cx q[24],q[0];
cx q[19],q[15];
rz(pi/2) q[15];
cx q[19],q[15];
cx q[6],q[3];
cx q[21],q[3];
cx q[15],q[3];
cx q[13],q[3];
rz(-pi/2) q[3];
cx q[13],q[3];
cx q[15],q[3];
cx q[21],q[3];
cx q[6],q[3];
cx q[18],q[3];
cx q[22],q[3];
cx q[9],q[3];
cx q[16],q[3];
cx q[4],q[3];
rz(5*pi/4) q[3];
cx q[4],q[3];
cx q[16],q[3];
cx q[9],q[3];
cx q[22],q[3];
cx q[18],q[3];
cx q[7],q[0];
cx q[11],q[7];
cx q[12],q[11];
cx q[17],q[15];
cx q[17],q[11];
cx q[17],q[5];
cx q[18],q[17];
cx q[18],q[15];
cx q[18],q[7];
cx q[19],q[16];
cx q[20],q[3];
cx q[21],q[4];
cx q[22],q[19];
cx q[22],q[16];
cx q[22],q[9];
cx q[22],q[2];
cx q[23],q[19];
cx q[23],q[16];
cx q[23],q[9];
cx q[23],q[2];
cx q[24],q[21];
cx q[22],q[23];
cx q[17],q[18];
cx q[18],q[17];
cx q[12],q[22];
cx q[12],q[18];
cx q[12],q[15];
cx q[12],q[13];
cx q[11],q[18];
cx q[11],q[17];
cx q[11],q[15];
cx q[17],q[11];
cx q[8],q[20];
cx q[8],q[10];
cx q[8],q[9];
cx q[7],q[18];
cx q[7],q[15];
cx q[7],q[11];
cx q[18],q[7];
cx q[5],q[8];
cx q[4],q[10];
cx q[2],q[19];
cx q[2],q[16];
cx q[2],q[9];
cx q[0],q[18];
cx q[0],q[11];
cx q[0],q[7];
cx q[11],q[0];
cx q[17],q[0];
cx q[21],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[23],q[0];
rz(3*pi/2) q[0];
cx q[23],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[21],q[0];
cx q[17],q[0];
cx q[17],q[8];
cx q[22],q[8];
rz(3*pi/2) q[8];
cx q[22],q[8];
cx q[17],q[8];
cx q[18],q[11];
rz(7*pi/4) q[11];
cx q[18],q[11];
cx q[20],q[18];
cx q[19],q[18];
cx q[21],q[18];
rz(7*pi/4) q[18];
cx q[21],q[18];
cx q[19],q[18];
cx q[20],q[18];
cx q[14],q[7];
cx q[8],q[7];
cx q[10],q[7];
cx q[9],q[7];
rz(3*pi/2) q[7];
cx q[9],q[7];
cx q[10],q[7];
cx q[8],q[7];
cx q[14],q[7];
cx q[18],q[17];
cx q[5],q[8];
cx q[0],q[1];
cx q[13],q[4];
cx q[10],q[4];
rz(3*pi/2) q[4];
cx q[10],q[4];
cx q[13],q[4];
rx(9*pi/4) q[6];
cx q[23],q[8];
cx q[9],q[8];
cx q[13],q[8];
rz(5*pi/4) q[8];
cx q[13],q[8];
cx q[9],q[8];
cx q[23],q[8];
cx q[19],q[1];
cx q[18],q[1];
rz(5*pi/4) q[1];
cx q[18],q[1];
cx q[19],q[1];
cx q[3],q[15];
cx q[3],q[8];
cx q[3],q[18];
rx(5*pi/4) q[3];
cx q[3],q[18];
cx q[3],q[8];
cx q[3],q[15];
cx q[0],q[22];
cx q[0],q[21];
cx q[0],q[3];
cx q[0],q[14];
rx(pi/4) q[0];
cx q[0],q[14];
cx q[0],q[3];
cx q[0],q[21];
cx q[0],q[22];
cx q[7],q[1];
cx q[7],q[0];
cx q[11],q[9];
cx q[13],q[11];
cx q[13],q[1];
cx q[16],q[11];
cx q[16],q[1];
cx q[17],q[9];
cx q[22],q[20];
cx q[22],q[10];
cx q[22],q[8];
cx q[24],q[21];
cx q[19],q[23];
cx q[16],q[24];
cx q[16],q[17];
cx q[14],q[24];
cx q[13],q[24];
cx q[13],q[17];
cx q[13],q[16];
cx q[11],q[17];
cx q[8],q[20];
cx q[8],q[10];
cx q[6],q[22];
cx q[5],q[23];
cx q[5],q[19];
cx q[1],q[24];
cx q[0],q[24];
cx q[0],q[18];
cx q[0],q[1];
cx q[21],q[3];
cx q[4],q[3];
cx q[18],q[3];
cx q[24],q[3];
rz(5*pi/4) q[3];
cx q[24],q[3];
cx q[18],q[3];
cx q[4],q[3];
cx q[21],q[3];
cx q[6],q[14];
cx q[6],q[12];
rx(pi/4) q[6];
cx q[6],q[12];
cx q[6],q[14];
cx q[22],q[3];
cx q[10],q[3];
rz(3*pi/2) q[3];
cx q[10],q[3];
cx q[22],q[3];
cx q[1],q[22];
cx q[1],q[17];
cx q[1],q[11];
rx(7*pi/4) q[1];
cx q[1],q[11];
cx q[1],q[17];
cx q[1],q[22];
cx q[14],q[1];
rz(5*pi/4) q[1];
cx q[14],q[1];
cx q[16],q[10];
cx q[16],q[24];
cx q[14],q[24];
cx q[6],q[22];
cx q[9],q[2];
cx q[3],q[2];
rz(7*pi/4) q[2];
cx q[3],q[2];
cx q[9],q[2];
cx q[15],q[19];
cx q[15],q[18];
rx(3*pi/4) q[15];
cx q[15],q[18];
cx q[15],q[19];
cx q[11],q[6];
cx q[23],q[6];
rz(7*pi/4) q[6];
cx q[23],q[6];
cx q[11],q[6];
cx q[6],q[16];
cx q[6],q[18];
rx(3*pi/4) q[6];
cx q[6],q[18];
cx q[6],q[16];
cx q[13],q[12];
cx q[24],q[12];
cx q[17],q[12];
rz(3*pi/2) q[12];
cx q[17],q[12];
cx q[24],q[12];
cx q[13],q[12];
cx q[20],q[10];
cx q[23],q[10];
cx q[22],q[10];
cx q[21],q[10];
rz(5*pi/4) q[10];
cx q[21],q[10];
cx q[22],q[10];
cx q[23],q[10];
cx q[20],q[10];
cx q[7],q[5];
cx q[11],q[7];
cx q[11],q[5];
cx q[16],q[11];
cx q[16],q[10];
cx q[16],q[7];
cx q[16],q[5];
cx q[16],q[1];
cx q[17],q[9];
cx q[18],q[17];
cx q[19],q[16];
cx q[19],q[2];
cx q[20],q[3];
cx q[22],q[8];
cx q[23],q[2];
cx q[22],q[23];
cx q[19],q[24];
cx q[19],q[23];
cx q[16],q[24];
cx q[13],q[16];
cx q[12],q[23];
cx q[12],q[22];
cx q[8],q[9];
cx q[7],q[19];
cx q[7],q[16];
cx q[5],q[19];
cx q[2],q[19];
cx q[2],q[9];
cx q[1],q[24];
cx q[0],q[11];
cx q[7],q[5];
cx q[12],q[5];
cx q[17],q[16];
cx q[17],q[5];
cx q[18],q[17];
cx q[18],q[16];
cx q[18],q[5];
cx q[21],q[10];
cx q[21],q[4];
cx q[12],q[17];
cx q[12],q[16];
cx q[7],q[16];
cx q[5],q[16];
cx q[4],q[10];
cx q[11],q[3];
cx q[5],q[3];
cx q[18],q[3];
cx q[7],q[3];
rz(pi) q[3];
cx q[7],q[3];
cx q[18],q[3];
cx q[5],q[3];
cx q[11],q[3];
cx q[11],q[8];
rz(pi) q[8];
cx q[11],q[8];
rz(pi) q[6];
cx q[18],q[17];
rz(pi) q[17];
cx q[18],q[17];
cx q[19],q[18];
rz(pi) q[18];
cx q[19],q[18];
cx q[13],q[10];
rz(pi) q[10];
cx q[13],q[10];
rz(pi) q[8];
cx q[21],q[10];
rz(pi) q[10];
cx q[21],q[10];
cx q[15],q[1];
rz(pi) q[1];
cx q[15],q[1];
cx q[12],q[22];
rx(pi) q[12];
cx q[12],q[22];
cx q[12],q[13];
cx q[7],q[15];
cx q[24],q[18];
rz(pi) q[18];
cx q[24],q[18];
rz(pi) q[18];
rx(pi) q[13];
rx(pi) q[16];
cx q[3],q[11];
rx(pi) q[3];
cx q[3],q[11];
cx q[14],q[22];
rx(pi) q[14];
cx q[14],q[22];
cx q[1],q[23];
rx(pi) q[1];
cx q[1],q[23];
cx q[24],q[0];
cx q[5],q[0];
cx q[13],q[0];
cx q[4],q[0];
cx q[23],q[0];
rz(pi/4) q[0];
cx q[23],q[0];
cx q[4],q[0];
cx q[13],q[0];
cx q[5],q[0];
cx q[24],q[0];
cx q[19],q[15];
rz(pi/2) q[15];
cx q[19],q[15];
cx q[6],q[3];
cx q[21],q[3];
cx q[15],q[3];
cx q[13],q[3];
rz(-pi/2) q[3];
cx q[13],q[3];
cx q[15],q[3];
cx q[21],q[3];
cx q[6],q[3];
cx q[18],q[3];
cx q[22],q[3];
cx q[9],q[3];
cx q[16],q[3];
cx q[4],q[3];
rz(5*pi/4) q[3];
cx q[4],q[3];
cx q[16],q[3];
cx q[9],q[3];
cx q[22],q[3];
cx q[18],q[3];
cx q[7],q[0];
cx q[11],q[7];
cx q[12],q[11];
cx q[17],q[15];
cx q[17],q[11];
cx q[17],q[5];
cx q[18],q[17];
cx q[18],q[15];
cx q[18],q[7];
cx q[19],q[16];
cx q[20],q[3];
cx q[21],q[4];
cx q[22],q[19];
cx q[22],q[16];
cx q[22],q[9];
cx q[22],q[2];
cx q[23],q[19];
cx q[23],q[16];
cx q[23],q[9];
cx q[23],q[2];
cx q[24],q[21];
cx q[22],q[23];
cx q[17],q[18];
cx q[18],q[17];
cx q[12],q[22];
cx q[12],q[18];
cx q[12],q[15];
cx q[12],q[13];
cx q[11],q[18];
cx q[11],q[17];
cx q[11],q[15];
cx q[17],q[11];
cx q[8],q[20];
cx q[8],q[10];
cx q[8],q[9];
cx q[7],q[18];
cx q[7],q[15];
cx q[7],q[11];
cx q[18],q[7];
cx q[5],q[8];
cx q[4],q[10];
cx q[2],q[19];
cx q[2],q[16];
cx q[2],q[9];
cx q[0],q[18];
cx q[0],q[11];
cx q[0],q[7];
cx q[11],q[0];
cx q[17],q[0];
cx q[21],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[23],q[0];
rz(3*pi/2) q[0];
cx q[23],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[21],q[0];
cx q[17],q[0];
cx q[17],q[8];
cx q[22],q[8];
rz(3*pi/2) q[8];
cx q[22],q[8];
cx q[17],q[8];
cx q[18],q[11];
rz(7*pi/4) q[11];
cx q[18],q[11];
cx q[20],q[18];
cx q[19],q[18];
cx q[21],q[18];
rz(7*pi/4) q[18];
cx q[21],q[18];
cx q[19],q[18];
cx q[20],q[18];
cx q[14],q[7];
cx q[8],q[7];
cx q[10],q[7];
cx q[9],q[7];
rz(3*pi/2) q[7];
cx q[9],q[7];
cx q[10],q[7];
cx q[8],q[7];
cx q[14],q[7];
cx q[18],q[17];
cx q[5],q[8];
cx q[0],q[1];
cx q[13],q[4];
cx q[10],q[4];
rz(3*pi/2) q[4];
cx q[10],q[4];
cx q[13],q[4];
rx(9*pi/4) q[6];
cx q[23],q[8];
cx q[9],q[8];
cx q[13],q[8];
rz(5*pi/4) q[8];
cx q[13],q[8];
cx q[9],q[8];
cx q[23],q[8];
cx q[19],q[1];
cx q[18],q[1];
rz(5*pi/4) q[1];
cx q[18],q[1];
cx q[19],q[1];
cx q[3],q[15];
cx q[3],q[8];
cx q[3],q[18];
rx(5*pi/4) q[3];
cx q[3],q[18];
cx q[3],q[8];
cx q[3],q[15];
cx q[0],q[22];
cx q[0],q[21];
cx q[0],q[3];
cx q[0],q[14];
rx(pi/4) q[0];
cx q[0],q[14];
cx q[0],q[3];
cx q[0],q[21];
cx q[0],q[22];
cx q[7],q[1];
cx q[7],q[0];
cx q[11],q[9];
cx q[13],q[11];
cx q[13],q[1];
cx q[16],q[11];
cx q[16],q[1];
cx q[17],q[9];
cx q[22],q[20];
cx q[22],q[10];
cx q[22],q[8];
cx q[24],q[21];
cx q[19],q[23];
cx q[16],q[24];
cx q[16],q[17];
cx q[14],q[24];
cx q[13],q[24];
cx q[13],q[17];
cx q[13],q[16];
cx q[11],q[17];
cx q[8],q[20];
cx q[8],q[10];
cx q[6],q[22];
cx q[5],q[23];
cx q[5],q[19];
cx q[1],q[24];
cx q[0],q[24];
cx q[0],q[18];
cx q[0],q[1];
cx q[21],q[3];
cx q[4],q[3];
cx q[18],q[3];
cx q[24],q[3];
rz(5*pi/4) q[3];
cx q[24],q[3];
cx q[18],q[3];
cx q[4],q[3];
cx q[21],q[3];
cx q[6],q[14];
cx q[6],q[12];
rx(pi/4) q[6];
cx q[6],q[12];
cx q[6],q[14];
cx q[22],q[3];
cx q[10],q[3];
rz(3*pi/2) q[3];
cx q[10],q[3];
cx q[22],q[3];
cx q[1],q[22];
cx q[1],q[17];
cx q[1],q[11];
rx(7*pi/4) q[1];
cx q[1],q[11];
cx q[1],q[17];
cx q[1],q[22];
cx q[14],q[1];
rz(5*pi/4) q[1];
cx q[14],q[1];
cx q[16],q[10];
cx q[16],q[24];
cx q[14],q[24];
cx q[6],q[22];
cx q[9],q[2];
cx q[3],q[2];
rz(7*pi/4) q[2];
cx q[3],q[2];
cx q[9],q[2];
cx q[15],q[19];
cx q[15],q[18];
rx(3*pi/4) q[15];
cx q[15],q[18];
cx q[15],q[19];
cx q[11],q[6];
cx q[23],q[6];
rz(7*pi/4) q[6];
cx q[23],q[6];
cx q[11],q[6];
cx q[6],q[16];
cx q[6],q[18];
rx(3*pi/4) q[6];
cx q[6],q[18];
cx q[6],q[16];
cx q[13],q[12];
cx q[24],q[12];
cx q[17],q[12];
rz(3*pi/2) q[12];
cx q[17],q[12];
cx q[24],q[12];
cx q[13],q[12];
cx q[20],q[10];
cx q[23],q[10];
cx q[22],q[10];
cx q[21],q[10];
rz(5*pi/4) q[10];
cx q[21],q[10];
cx q[22],q[10];
cx q[23],q[10];
cx q[20],q[10];
cx q[7],q[5];
cx q[11],q[7];
cx q[11],q[5];
cx q[16],q[11];
cx q[16],q[10];
cx q[16],q[7];
cx q[16],q[5];
cx q[16],q[1];
cx q[17],q[9];
cx q[18],q[17];
cx q[19],q[16];
cx q[19],q[2];
cx q[20],q[3];
cx q[22],q[8];
cx q[23],q[2];
cx q[22],q[23];
cx q[19],q[24];
cx q[19],q[23];
cx q[16],q[24];
cx q[13],q[16];
cx q[12],q[23];
cx q[12],q[22];
cx q[8],q[9];
cx q[7],q[19];
cx q[7],q[16];
cx q[5],q[19];
cx q[2],q[19];
cx q[2],q[9];
cx q[1],q[24];
cx q[0],q[11];
cx q[7],q[5];
cx q[12],q[5];
cx q[17],q[16];
cx q[17],q[5];
cx q[18],q[17];
cx q[18],q[16];
cx q[18],q[5];
cx q[21],q[10];
cx q[21],q[4];
cx q[12],q[17];
cx q[12],q[16];
cx q[7],q[16];
cx q[5],q[16];
cx q[4],q[10];
cx q[11],q[3];
cx q[5],q[3];
cx q[18],q[3];
cx q[7],q[3];
rz(pi) q[3];
cx q[7],q[3];
cx q[18],q[3];
cx q[5],q[3];
cx q[11],q[3];
cx q[11],q[8];
rz(pi) q[8];
cx q[11],q[8];
rz(pi) q[6];
cx q[18],q[17];
rz(pi) q[17];
cx q[18],q[17];
cx q[19],q[18];
rz(pi) q[18];
cx q[19],q[18];
cx q[13],q[10];
rz(pi) q[10];
cx q[13],q[10];
rz(pi) q[8];
cx q[21],q[10];
rz(pi) q[10];
cx q[21],q[10];
cx q[15],q[1];
rz(pi) q[1];
cx q[15],q[1];
cx q[12],q[22];
rx(pi) q[12];
cx q[12],q[22];
cx q[12],q[13];
cx q[7],q[15];
cx q[24],q[18];
rz(pi) q[18];
cx q[24],q[18];
rz(pi) q[18];
rx(pi) q[13];
rx(pi) q[16];
cx q[3],q[11];
rx(pi) q[3];
cx q[3],q[11];
cx q[14],q[22];
rx(pi) q[14];
cx q[14],q[22];
cx q[1],q[23];
rx(pi) q[1];
cx q[1],q[23];
cx q[24],q[0];
cx q[5],q[0];
cx q[13],q[0];
cx q[4],q[0];
cx q[23],q[0];
rz(pi/4) q[0];
cx q[23],q[0];
cx q[4],q[0];
cx q[13],q[0];
cx q[5],q[0];
cx q[24],q[0];
cx q[19],q[15];
rz(pi/2) q[15];
cx q[19],q[15];
cx q[6],q[3];
cx q[21],q[3];
cx q[15],q[3];
cx q[13],q[3];
rz(-pi/2) q[3];
cx q[13],q[3];
cx q[15],q[3];
cx q[21],q[3];
cx q[6],q[3];
cx q[18],q[3];
cx q[22],q[3];
cx q[9],q[3];
cx q[16],q[3];
cx q[4],q[3];
rz(5*pi/4) q[3];
cx q[4],q[3];
cx q[16],q[3];
cx q[9],q[3];
cx q[22],q[3];
cx q[18],q[3];
cx q[7],q[0];
cx q[11],q[7];
cx q[12],q[11];
cx q[17],q[15];
cx q[17],q[11];
cx q[17],q[5];
cx q[18],q[17];
cx q[18],q[15];
cx q[18],q[7];
cx q[19],q[16];
cx q[20],q[3];
cx q[21],q[4];
cx q[22],q[19];
cx q[22],q[16];
cx q[22],q[9];
cx q[22],q[2];
cx q[23],q[19];
cx q[23],q[16];
cx q[23],q[9];
cx q[23],q[2];
cx q[24],q[21];
cx q[22],q[23];
cx q[17],q[18];
cx q[18],q[17];
cx q[12],q[22];
cx q[12],q[18];
cx q[12],q[15];
cx q[12],q[13];
cx q[11],q[18];
cx q[11],q[17];
cx q[11],q[15];
cx q[17],q[11];
cx q[8],q[20];
cx q[8],q[10];
cx q[8],q[9];
cx q[7],q[18];
cx q[7],q[15];
cx q[7],q[11];
cx q[18],q[7];
cx q[5],q[8];
cx q[4],q[10];
cx q[2],q[19];
cx q[2],q[16];
cx q[2],q[9];
cx q[0],q[18];
cx q[0],q[11];
cx q[0],q[7];
cx q[11],q[0];
cx q[17],q[0];
cx q[21],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[23],q[0];
rz(3*pi/2) q[0];
cx q[23],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[21],q[0];
cx q[17],q[0];
cx q[17],q[8];
cx q[22],q[8];
rz(3*pi/2) q[8];
cx q[22],q[8];
cx q[17],q[8];
cx q[18],q[11];
rz(7*pi/4) q[11];
cx q[18],q[11];
cx q[20],q[18];
cx q[19],q[18];
cx q[21],q[18];
rz(7*pi/4) q[18];
cx q[21],q[18];
cx q[19],q[18];
cx q[20],q[18];
cx q[14],q[7];
cx q[8],q[7];
cx q[10],q[7];
cx q[9],q[7];
rz(3*pi/2) q[7];
cx q[9],q[7];
cx q[10],q[7];
cx q[8],q[7];
cx q[14],q[7];
cx q[18],q[17];
cx q[5],q[8];
cx q[0],q[1];
cx q[13],q[4];
cx q[10],q[4];
rz(3*pi/2) q[4];
cx q[10],q[4];
cx q[13],q[4];
rx(9*pi/4) q[6];
cx q[23],q[8];
cx q[9],q[8];
cx q[13],q[8];
rz(5*pi/4) q[8];
cx q[13],q[8];
cx q[9],q[8];
cx q[23],q[8];
cx q[19],q[1];
cx q[18],q[1];
rz(5*pi/4) q[1];
cx q[18],q[1];
cx q[19],q[1];
cx q[3],q[15];
cx q[3],q[8];
cx q[3],q[18];
rx(5*pi/4) q[3];
cx q[3],q[18];
cx q[3],q[8];
cx q[3],q[15];
cx q[0],q[22];
cx q[0],q[21];
cx q[0],q[3];
cx q[0],q[14];
rx(pi/4) q[0];
cx q[0],q[14];
cx q[0],q[3];
cx q[0],q[21];
cx q[0],q[22];
cx q[7],q[1];
cx q[7],q[0];
cx q[11],q[9];
cx q[13],q[11];
cx q[13],q[1];
cx q[16],q[11];
cx q[16],q[1];
cx q[17],q[9];
cx q[22],q[20];
cx q[22],q[10];
cx q[22],q[8];
cx q[24],q[21];
cx q[19],q[23];
cx q[16],q[24];
cx q[16],q[17];
cx q[14],q[24];
cx q[13],q[24];
cx q[13],q[17];
cx q[13],q[16];
cx q[11],q[17];
cx q[8],q[20];
cx q[8],q[10];
cx q[6],q[22];
cx q[5],q[23];
cx q[5],q[19];
cx q[1],q[24];
cx q[0],q[24];
cx q[0],q[18];
cx q[0],q[1];
cx q[21],q[3];
cx q[4],q[3];
cx q[18],q[3];
cx q[24],q[3];
rz(5*pi/4) q[3];
cx q[24],q[3];
cx q[18],q[3];
cx q[4],q[3];
cx q[21],q[3];
cx q[6],q[14];
cx q[6],q[12];
rx(pi/4) q[6];
cx q[6],q[12];
cx q[6],q[14];
cx q[22],q[3];
cx q[10],q[3];
rz(3*pi/2) q[3];
cx q[10],q[3];
cx q[22],q[3];
cx q[1],q[22];
cx q[1],q[17];
cx q[1],q[11];
rx(7*pi/4) q[1];
cx q[1],q[11];
cx q[1],q[17];
cx q[1],q[22];
cx q[14],q[1];
rz(5*pi/4) q[1];
cx q[14],q[1];
cx q[16],q[10];
cx q[16],q[24];
cx q[14],q[24];
cx q[6],q[22];
cx q[9],q[2];
cx q[3],q[2];
rz(7*pi/4) q[2];
cx q[3],q[2];
cx q[9],q[2];
cx q[15],q[19];
cx q[15],q[18];
rx(3*pi/4) q[15];
cx q[15],q[18];
cx q[15],q[19];
cx q[11],q[6];
cx q[23],q[6];
rz(7*pi/4) q[6];
cx q[23],q[6];
cx q[11],q[6];
cx q[6],q[16];
cx q[6],q[18];
rx(3*pi/4) q[6];
cx q[6],q[18];
cx q[6],q[16];
cx q[13],q[12];
cx q[24],q[12];
cx q[17],q[12];
rz(3*pi/2) q[12];
cx q[17],q[12];
cx q[24],q[12];
cx q[13],q[12];
cx q[20],q[10];
cx q[23],q[10];
cx q[22],q[10];
cx q[21],q[10];
rz(5*pi/4) q[10];
cx q[21],q[10];
cx q[22],q[10];
cx q[23],q[10];
cx q[20],q[10];
cx q[7],q[5];
cx q[11],q[7];
cx q[11],q[5];
cx q[16],q[11];
cx q[16],q[10];
cx q[16],q[7];
cx q[16],q[5];
cx q[16],q[1];
cx q[17],q[9];
cx q[18],q[17];
cx q[19],q[16];
cx q[19],q[2];
cx q[20],q[3];
cx q[22],q[8];
cx q[23],q[2];
cx q[22],q[23];
cx q[19],q[24];
cx q[19],q[23];
cx q[16],q[24];
cx q[13],q[16];
cx q[12],q[23];
cx q[12],q[22];
cx q[8],q[9];
cx q[7],q[19];
cx q[7],q[16];
cx q[5],q[19];
cx q[2],q[19];
cx q[2],q[9];
cx q[1],q[24];
cx q[0],q[11];
cx q[7],q[5];
cx q[12],q[5];
cx q[17],q[16];
cx q[17],q[5];
cx q[18],q[17];
cx q[18],q[16];
cx q[18],q[5];
cx q[21],q[10];
cx q[21],q[4];
cx q[12],q[17];
cx q[12],q[16];
cx q[7],q[16];
cx q[5],q[16];
cx q[4],q[10];
cx q[11],q[3];
cx q[5],q[3];
cx q[18],q[3];
cx q[7],q[3];
rz(pi) q[3];
cx q[7],q[3];
cx q[18],q[3];
cx q[5],q[3];
cx q[11],q[3];
cx q[11],q[8];
rz(pi) q[8];
cx q[11],q[8];
rz(pi) q[6];
cx q[18],q[17];
rz(pi) q[17];
cx q[18],q[17];
cx q[19],q[18];
rz(pi) q[18];
cx q[19],q[18];
cx q[13],q[10];
rz(pi) q[10];
cx q[13],q[10];
rz(pi) q[8];
cx q[21],q[10];
rz(pi) q[10];
cx q[21],q[10];
cx q[15],q[1];
rz(pi) q[1];
cx q[15],q[1];
cx q[12],q[22];
rx(pi) q[12];
cx q[12],q[22];
cx q[12],q[13];
cx q[7],q[15];
cx q[24],q[18];
rz(pi) q[18];
cx q[24],q[18];
rz(pi) q[18];
rx(pi) q[13];
rx(pi) q[16];
cx q[3],q[11];
rx(pi) q[3];
cx q[3],q[11];
cx q[14],q[22];
rx(pi) q[14];
cx q[14],q[22];
cx q[1],q[23];
rx(pi) q[1];
cx q[1],q[23];
cx q[24],q[0];
cx q[5],q[0];
cx q[13],q[0];
cx q[4],q[0];
cx q[23],q[0];
rz(pi/4) q[0];
cx q[23],q[0];
cx q[4],q[0];
cx q[13],q[0];
cx q[5],q[0];
cx q[24],q[0];
cx q[19],q[15];
rz(pi/2) q[15];
cx q[19],q[15];
cx q[6],q[3];
cx q[21],q[3];
cx q[15],q[3];
cx q[13],q[3];
rz(-pi/2) q[3];
cx q[13],q[3];
cx q[15],q[3];
cx q[21],q[3];
cx q[6],q[3];
cx q[18],q[3];
cx q[22],q[3];
cx q[9],q[3];
cx q[16],q[3];
cx q[4],q[3];
rz(5*pi/4) q[3];
cx q[4],q[3];
cx q[16],q[3];
cx q[9],q[3];
cx q[22],q[3];
cx q[18],q[3];
cx q[7],q[0];
cx q[11],q[7];
cx q[12],q[11];
cx q[17],q[15];
cx q[17],q[11];
cx q[17],q[5];
cx q[18],q[17];
cx q[18],q[15];
cx q[18],q[7];
cx q[19],q[16];
cx q[20],q[3];
cx q[21],q[4];
cx q[22],q[19];
cx q[22],q[16];
cx q[22],q[9];
cx q[22],q[2];
cx q[23],q[19];
cx q[23],q[16];
cx q[23],q[9];
cx q[23],q[2];
cx q[24],q[21];
cx q[22],q[23];
cx q[17],q[18];
cx q[18],q[17];
cx q[12],q[22];
cx q[12],q[18];
cx q[12],q[15];
cx q[12],q[13];
cx q[11],q[18];
cx q[11],q[17];
cx q[11],q[15];
cx q[17],q[11];
cx q[8],q[20];
cx q[8],q[10];
cx q[8],q[9];
cx q[7],q[18];
cx q[7],q[15];
cx q[7],q[11];
cx q[18],q[7];
cx q[5],q[8];
cx q[4],q[10];
cx q[2],q[19];
cx q[2],q[16];
cx q[2],q[9];
cx q[0],q[18];
cx q[0],q[11];
cx q[0],q[7];
cx q[11],q[0];
cx q[17],q[0];
cx q[21],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[23],q[0];
rz(3*pi/2) q[0];
cx q[23],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[21],q[0];
cx q[17],q[0];
cx q[17],q[8];
cx q[22],q[8];
rz(3*pi/2) q[8];
cx q[22],q[8];
cx q[17],q[8];
cx q[18],q[11];
rz(7*pi/4) q[11];
cx q[18],q[11];
cx q[20],q[18];
cx q[19],q[18];
cx q[21],q[18];
rz(7*pi/4) q[18];
cx q[21],q[18];
cx q[19],q[18];
cx q[20],q[18];
cx q[14],q[7];
cx q[8],q[7];
cx q[10],q[7];
cx q[9],q[7];
rz(3*pi/2) q[7];
cx q[9],q[7];
cx q[10],q[7];
cx q[8],q[7];
cx q[14],q[7];
cx q[18],q[17];
cx q[5],q[8];
cx q[0],q[1];
cx q[13],q[4];
cx q[10],q[4];
rz(3*pi/2) q[4];
cx q[10],q[4];
cx q[13],q[4];
rx(9*pi/4) q[6];
cx q[23],q[8];
cx q[9],q[8];
cx q[13],q[8];
rz(5*pi/4) q[8];
cx q[13],q[8];
cx q[9],q[8];
cx q[23],q[8];
cx q[19],q[1];
cx q[18],q[1];
rz(5*pi/4) q[1];
cx q[18],q[1];
cx q[19],q[1];
cx q[3],q[15];
cx q[3],q[8];
cx q[3],q[18];
rx(5*pi/4) q[3];
cx q[3],q[18];
cx q[3],q[8];
cx q[3],q[15];
cx q[0],q[22];
cx q[0],q[21];
cx q[0],q[3];
cx q[0],q[14];
rx(pi/4) q[0];
cx q[0],q[14];
cx q[0],q[3];
cx q[0],q[21];
cx q[0],q[22];
cx q[7],q[1];
cx q[7],q[0];
cx q[11],q[9];
cx q[13],q[11];
cx q[13],q[1];
cx q[16],q[11];
cx q[16],q[1];
cx q[17],q[9];
cx q[22],q[20];
cx q[22],q[10];
cx q[22],q[8];
cx q[24],q[21];
cx q[19],q[23];
cx q[16],q[24];
cx q[16],q[17];
cx q[14],q[24];
cx q[13],q[24];
cx q[13],q[17];
cx q[13],q[16];
cx q[11],q[17];
cx q[8],q[20];
cx q[8],q[10];
cx q[6],q[22];
cx q[5],q[23];
cx q[5],q[19];
cx q[1],q[24];
cx q[0],q[24];
cx q[0],q[18];
cx q[0],q[1];
cx q[21],q[3];
cx q[4],q[3];
cx q[18],q[3];
cx q[24],q[3];
rz(5*pi/4) q[3];
cx q[24],q[3];
cx q[18],q[3];
cx q[4],q[3];
cx q[21],q[3];
cx q[6],q[14];
cx q[6],q[12];
rx(pi/4) q[6];
cx q[6],q[12];
cx q[6],q[14];
cx q[22],q[3];
cx q[10],q[3];
rz(3*pi/2) q[3];
cx q[10],q[3];
cx q[22],q[3];
cx q[1],q[22];
cx q[1],q[17];
cx q[1],q[11];
rx(7*pi/4) q[1];
cx q[1],q[11];
cx q[1],q[17];
cx q[1],q[22];
cx q[14],q[1];
rz(5*pi/4) q[1];
cx q[14],q[1];
cx q[16],q[10];
cx q[16],q[24];
cx q[14],q[24];
cx q[6],q[22];
cx q[9],q[2];
cx q[3],q[2];
rz(7*pi/4) q[2];
cx q[3],q[2];
cx q[9],q[2];
cx q[15],q[19];
cx q[15],q[18];
rx(3*pi/4) q[15];
cx q[15],q[18];
cx q[15],q[19];
cx q[11],q[6];
cx q[23],q[6];
rz(7*pi/4) q[6];
cx q[23],q[6];
cx q[11],q[6];
cx q[6],q[16];
cx q[6],q[18];
rx(3*pi/4) q[6];
cx q[6],q[18];
cx q[6],q[16];
cx q[13],q[12];
cx q[24],q[12];
cx q[17],q[12];
rz(3*pi/2) q[12];
cx q[17],q[12];
cx q[24],q[12];
cx q[13],q[12];
cx q[20],q[10];
cx q[23],q[10];
cx q[22],q[10];
cx q[21],q[10];
rz(5*pi/4) q[10];
cx q[21],q[10];
cx q[22],q[10];
cx q[23],q[10];
cx q[20],q[10];
cx q[7],q[5];
cx q[11],q[7];
cx q[11],q[5];
cx q[16],q[11];
cx q[16],q[10];
cx q[16],q[7];
cx q[16],q[5];
cx q[16],q[1];
cx q[17],q[9];
cx q[18],q[17];
cx q[19],q[16];
cx q[19],q[2];
cx q[20],q[3];
cx q[22],q[8];
cx q[23],q[2];
cx q[22],q[23];
cx q[19],q[24];
cx q[19],q[23];
cx q[16],q[24];
cx q[13],q[16];
cx q[12],q[23];
cx q[12],q[22];
cx q[8],q[9];
cx q[7],q[19];
cx q[7],q[16];
cx q[5],q[19];
cx q[2],q[19];
cx q[2],q[9];
cx q[1],q[24];
cx q[0],q[11];
cx q[7],q[5];
cx q[12],q[5];
cx q[17],q[16];
cx q[17],q[5];
cx q[18],q[17];
cx q[18],q[16];
cx q[18],q[5];
cx q[21],q[10];
cx q[21],q[4];
cx q[12],q[17];
cx q[12],q[16];
cx q[7],q[16];
cx q[5],q[16];
cx q[4],q[10];
cx q[11],q[3];
cx q[5],q[3];
cx q[18],q[3];
cx q[7],q[3];
rz(pi) q[3];
cx q[7],q[3];
cx q[18],q[3];
cx q[5],q[3];
cx q[11],q[3];
cx q[11],q[8];
rz(pi) q[8];
cx q[11],q[8];
rz(pi) q[6];
cx q[18],q[17];
rz(pi) q[17];
cx q[18],q[17];
cx q[19],q[18];
rz(pi) q[18];
cx q[19],q[18];
cx q[13],q[10];
rz(pi) q[10];
cx q[13],q[10];
rz(pi) q[8];
cx q[21],q[10];
rz(pi) q[10];
cx q[21],q[10];
cx q[15],q[1];
rz(pi) q[1];
cx q[15],q[1];
cx q[12],q[22];
rx(pi) q[12];
cx q[12],q[22];
cx q[12],q[13];
cx q[7],q[15];
cx q[24],q[18];
rz(pi) q[18];
cx q[24],q[18];
rz(pi) q[18];
rx(pi) q[13];
rx(pi) q[16];
cx q[3],q[11];
rx(pi) q[3];
cx q[3],q[11];
cx q[14],q[22];
rx(pi) q[14];
cx q[14],q[22];
cx q[1],q[23];
rx(pi) q[1];
cx q[1],q[23];
cx q[24],q[0];
cx q[5],q[0];
cx q[13],q[0];
cx q[4],q[0];
cx q[23],q[0];
rz(pi/4) q[0];
cx q[23],q[0];
cx q[4],q[0];
cx q[13],q[0];
cx q[5],q[0];
cx q[24],q[0];
cx q[19],q[15];
rz(pi/2) q[15];
cx q[19],q[15];
cx q[6],q[3];
cx q[21],q[3];
cx q[15],q[3];
cx q[13],q[3];
rz(-pi/2) q[3];
cx q[13],q[3];
cx q[15],q[3];
cx q[21],q[3];
cx q[6],q[3];
cx q[18],q[3];
cx q[22],q[3];
cx q[9],q[3];
cx q[16],q[3];
cx q[4],q[3];
rz(5*pi/4) q[3];
cx q[4],q[3];
cx q[16],q[3];
cx q[9],q[3];
cx q[22],q[3];
cx q[18],q[3];
cx q[7],q[0];
cx q[11],q[7];
cx q[12],q[11];
cx q[17],q[15];
cx q[17],q[11];
cx q[17],q[5];
cx q[18],q[17];
cx q[18],q[15];
cx q[18],q[7];
cx q[19],q[16];
cx q[20],q[3];
cx q[21],q[4];
cx q[22],q[19];
cx q[22],q[16];
cx q[22],q[9];
cx q[22],q[2];
cx q[23],q[19];
cx q[23],q[16];
cx q[23],q[9];
cx q[23],q[2];
cx q[24],q[21];
cx q[22],q[23];
cx q[17],q[18];
cx q[18],q[17];
cx q[12],q[22];
cx q[12],q[18];
cx q[12],q[15];
cx q[12],q[13];
cx q[11],q[18];
cx q[11],q[17];
cx q[11],q[15];
cx q[17],q[11];
cx q[8],q[20];
cx q[8],q[10];
cx q[8],q[9];
cx q[7],q[18];
cx q[7],q[15];
cx q[7],q[11];
cx q[18],q[7];
cx q[5],q[8];
cx q[4],q[10];
cx q[2],q[19];
cx q[2],q[16];
cx q[2],q[9];
cx q[0],q[18];
cx q[0],q[11];
cx q[0],q[7];
cx q[11],q[0];
cx q[17],q[0];
cx q[21],q[0];
cx q[16],q[0];
cx q[10],q[0];
cx q[23],q[0];
rz(3*pi/2) q[0];
cx q[23],q[0];
cx q[10],q[0];
cx q[16],q[0];
cx q[21],q[0];
cx q[17],q[0];
cx q[17],q[8];
cx q[22],q[8];
rz(3*pi/2) q[8];
cx q[22],q[8];
cx q[17],q[8];
cx q[18],q[11];
rz(7*pi/4) q[11];
cx q[18],q[11];
cx q[20],q[18];
cx q[19],q[18];
cx q[21],q[18];
rz(7*pi/4) q[18];
cx q[21],q[18];
cx q[19],q[18];
cx q[20],q[18];
cx q[14],q[7];
cx q[8],q[7];
cx q[10],q[7];
cx q[9],q[7];
rz(3*pi/2) q[7];
cx q[9],q[7];
cx q[10],q[7];
cx q[8],q[7];
cx q[14],q[7];
cx q[18],q[17];
cx q[5],q[8];
cx q[0],q[1];
cx q[13],q[4];
cx q[10],q[4];
rz(3*pi/2) q[4];
cx q[10],q[4];
cx q[13],q[4];
rx(9*pi/4) q[6];
cx q[23],q[8];
cx q[9],q[8];
cx q[13],q[8];
rz(5*pi/4) q[8];
cx q[13],q[8];
cx q[9],q[8];
cx q[23],q[8];
cx q[19],q[1];
cx q[18],q[1];
rz(5*pi/4) q[1];
cx q[18],q[1];
cx q[19],q[1];
cx q[3],q[15];
cx q[3],q[8];
cx q[3],q[18];
rx(5*pi/4) q[3];
cx q[3],q[18];
cx q[3],q[8];
cx q[3],q[15];
cx q[0],q[22];
cx q[0],q[21];
cx q[0],q[3];
cx q[0],q[14];
rx(pi/4) q[0];
cx q[0],q[14];
cx q[0],q[3];
cx q[0],q[21];
cx q[0],q[22];
cx q[7],q[1];
cx q[7],q[0];
cx q[11],q[9];
cx q[13],q[11];
cx q[13],q[1];
cx q[16],q[11];
cx q[16],q[1];
cx q[17],q[9];
cx q[22],q[20];
cx q[22],q[10];
cx q[22],q[8];
cx q[24],q[21];
cx q[19],q[23];
cx q[16],q[24];
cx q[16],q[17];
cx q[14],q[24];
cx q[13],q[24];
cx q[13],q[17];
cx q[13],q[16];
cx q[11],q[17];
cx q[8],q[20];
cx q[8],q[10];
cx q[6],q[22];
cx q[5],q[23];
cx q[5],q[19];
cx q[1],q[24];
cx q[0],q[24];
cx q[0],q[18];
cx q[0],q[1];
cx q[21],q[3];
cx q[4],q[3];
cx q[18],q[3];
cx q[24],q[3];
rz(5*pi/4) q[3];
cx q[24],q[3];
cx q[18],q[3];
cx q[4],q[3];
cx q[21],q[3];
cx q[6],q[14];
cx q[6],q[12];
rx(pi/4) q[6];
cx q[6],q[12];
cx q[6],q[14];
cx q[22],q[3];
cx q[10],q[3];
rz(3*pi/2) q[3];
cx q[10],q[3];
cx q[22],q[3];
cx q[1],q[22];
cx q[1],q[17];
cx q[1],q[11];
rx(7*pi/4) q[1];
cx q[1],q[11];
cx q[1],q[17];
cx q[1],q[22];
cx q[14],q[1];
rz(5*pi/4) q[1];
cx q[14],q[1];
cx q[16],q[10];
cx q[16],q[24];
cx q[14],q[24];
cx q[6],q[22];
cx q[9],q[2];
cx q[3],q[2];
rz(7*pi/4) q[2];
cx q[3],q[2];
cx q[9],q[2];
cx q[15],q[19];
cx q[15],q[18];
rx(3*pi/4) q[15];
cx q[15],q[18];
cx q[15],q[19];
cx q[11],q[6];
cx q[23],q[6];
rz(7*pi/4) q[6];
cx q[23],q[6];
cx q[11],q[6];
cx q[6],q[16];
cx q[6],q[18];
rx(3*pi/4) q[6];
cx q[6],q[18];
cx q[6],q[16];
cx q[13],q[12];
cx q[24],q[12];
cx q[17],q[12];
rz(3*pi/2) q[12];
cx q[17],q[12];
cx q[24],q[12];
cx q[13],q[12];
cx q[20],q[10];
cx q[23],q[10];
cx q[22],q[10];
cx q[21],q[10];
rz(5*pi/4) q[10];
cx q[21],q[10];
cx q[22],q[10];
cx q[23],q[10];
cx q[20],q[10];
cx q[7],q[5];
cx q[11],q[7];
cx q[11],q[5];
cx q[16],q[11];
cx q[16],q[10];
cx q[16],q[7];
cx q[16],q[5];
cx q[16],q[1];
cx q[17],q[9];
cx q[18],q[17];
cx q[19],q[16];
cx q[19],q[2];
cx q[20],q[3];
cx q[22],q[8];
cx q[23],q[2];
cx q[22],q[23];
cx q[19],q[24];
cx q[19],q[23];
cx q[16],q[24];
cx q[13],q[16];
cx q[12],q[23];
cx q[12],q[22];
cx q[8],q[9];
cx q[7],q[19];
cx q[7],q[16];
cx q[5],q[19];
cx q[2],q[19];
cx q[2],q[9];
cx q[1],q[24];
cx q[0],q[11];
cx q[4],q[0];
cx q[8],q[2];
cx q[10],q[4];
cx q[10],q[0];
cx q[11],q[3];
cx q[12],q[5];
cx q[14],q[5];
cx q[14],q[3];
cx q[15],q[9];
cx q[15],q[1];
cx q[17],q[12];
cx q[17],q[5];
cx q[19],q[7];
cx q[22],q[14];
cx q[22],q[5];
cx q[22],q[3];
cx q[23],q[1];
cx q[24],q[16];
cx q[18],q[24];
cx q[18],q[20];
cx q[18],q[19];
cx q[17],q[22];
cx q[14],q[17];
cx q[9],q[17];
cx q[8],q[11];
cx q[5],q[14];
cx q[4],q[21];
cx q[4],q[13];
cx q[4],q[9];
cx q[3],q[5];
cx q[2],q[4];
cx q[1],q[23];
cx q[1],q[16];
cx q[1],q[2];
cx q[0],q[3];
cx q[0],q[2];
