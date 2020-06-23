knnfunction = myknn(A,B,n,r,k)
p =size(A,1);
q =size(B,1);
for z= 1:q
    D =abs(minko_dist123(A,B(z,:),n,r));
    (F,I)= sort(D);
    G= I(1:k,1);
    G<=50;
    m=sum(G<=50);
    G<=100;
    s= sum(G<=100);
    G<=150;
    l= sum(G<=150);
    if (m>s) && (m>l) 
        H(z,1) =1;
    elseif (s>m) && (s>l)
        H(z,1) =2;
    else (l>m) && (l>s)
        H()z,1) =3;
    end
    
        
        
            
        