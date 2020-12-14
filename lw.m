
pkg load optim;

load quasar_train.csv;
lambdas = quasar_train(1, :)';
train_qso = quasar_train(2:end, :);
load quasar_test.csv;
test_qso = quasar_test(2:end, :);


Xo=[ones(rows(lambdas),1),lambdas];

Yo=train_qso(1,:)';

Yp=test_qso(1,:)';


minX = min(Xo(:,2));
maxX = max(Xo(:,2));
mx = 2/(maxX-minX);
bx = 1-mx*maxX;

NM=[1 bx; 0 mx];
X=Xo*NM; 



minY=min(Yo);
maxY=max(Yo);
my = 2/(maxY-minY);
by = 1-my*maxY;

Y = my*Yo + by;

imy = 1/my;
iby = -by/my;



function res=J(theta,X,Y)
  D=(X*theta'-Y*ones(1,rows(theta)));
  res=0.5*sum(D.*D,1)';
endfunction;


function res=gradJ(theta,X,Y)
  res=(X'*(X*theta'-Y*ones(1,rows(theta))))';
endfunction;



  figure(1);
  hold off;
  plot(Xo(:,2),Yo,"*b");
  hold on;
  
  t=(X'*X)^(-1)*X'*Y;
  
  text (1400, 4, "óptimo thetha0");
  text (1400, 3.5,num2str (t(1)));
  
  
  text (1400, 3, "óptimo thetha1");
  text (1400, 2.5,num2str (t(2)));
  
  
  xlabel("lambda");
  ylabel("flux");
  
 
  printf("óptimo theta_0= %d óptimo theta_1= %d",t(1),t(2));
  fflush(stdout);
  
  

  lamb=linspace(min(Xo(:,2)),max(Xo(:,2)),5);


  flux=t(2)*imy*mx*lamb + (imy*t(2)*bx + imy*t(1)+iby);
  

  

  plot(lamb,flux,'g',"linewidth",3);

  
  axis([minX maxX minY maxY]);  
  
  
  
  ##########Regresión ponderada localmente
  
  W=zeros(length(X),length(X));
  the=zeros(length(X),5,2);
  T=5;
  Out=0;
  
  for (k=[1:length(X)])
    
      for (i=[1:length(X)])
        
        W(i,i)=exp(-(norm(Xo(k,:)-Xo(i,:))^2)/(2*T^2));
        
      endfor
        
      the(k,1,:)=(X'*W*X)^(-1)*X'*W*Y;
      
        
      for (j=[0:3])
        
          for (i=[1:length(X)])
            
            W(i,i)=exp(-(norm(Xo(k,:)-Xo(i,:))^2)/(2*10^(2*j)));
            
          endfor  
          
          the(k,j+2,:)=(X'*W*X)^(-1)*X'*W*Y;
      endfor
      
      
       
      Out1(k)=the(k,2,2)*imy*X(k,2) + imy*the(k,2,1)+iby;
      Out5(k)=the(k,1,2)*imy*X(k,2) + imy*the(k,1,1)+iby; 
      Out10(k)=the(k,3,2)*imy*X(k,2) + imy*the(k,3,1)+iby;
      Out100(k)=the(k,4,2)*imy*X(k,2) + imy*the(k,4,1)+iby;
      Out1000(k)=the(k,5,2)*imy*X(k,2) + imy*the(k,5,1)+iby;
      


  endfor

  
  figure(2);
  hold off;
  plot(Xo(:,2),Yo,"o");
  hold on;
  plot(Xo(:,2),Yp,"*");
  plot(Xo(:,2),Out1,"linewidth",2);
  plot(Xo(:,2),Out5,"linewidth",2);
  plot(Xo(:,2),Out10,"linewidth",2);
  plot(Xo(:,2),Out100,"linewidth",2);
  plot(Xo(:,2),Out1000,"linewidth",2);
  
  
  
  legend('Datos crudos','Datos prueba','t=1','t=5','t=10','t=100','t=1000');
  axis([minX maxX minY maxY]);  
  xlabel("lambda");
  ylabel("flux");

