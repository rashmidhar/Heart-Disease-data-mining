function k =tdistance(X(:,1),X(:,2))
Ex =mean(X(:,1));
Ey =mean(X(:,2);
s =Ex-Ey;
u =X(:,1)-X(:,2);
k =(abs(s))/(std(u));
end
