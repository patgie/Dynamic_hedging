%load('date_1621-1650.mat')
%load('allS_new.mat')

prices=[750:1:850];
previous_date_model_hedge_values_lookback=zeros(size(maturities,1)-1,length(prices));
previous_date_model_hedge_values_straddle=zeros(size(maturities,1)-1,length(prices));
model_hedge_values_lookback=zeros(size(maturities,1)-1,length(prices));
model_hedge_values_straddle=zeros(size(maturities,1)-1,length(prices));

for i=1:size(maturities,1)-1
	read_string_lookback = strcat('Hedges_lookback_seed_456_cal_day_',string(i-1),'.csv');
	read_string_straddle = strcat('Hedges_straddle_seed_456_cal_day_',string(i-1),'.csv');
	hedge_values_lookback_temp=csvread(read_string_lookback, 1, 1);
	hedge_values_straddle_temp=csvread(read_string_straddle, 1, 1);
    if i<size(maturities,1)-1
        previous_date_model_hedge_values_lookback(i+1,:)=hedge_values_lookback_temp(:,2);
        previous_date_model_hedge_values_straddle(i+1,:)=hedge_values_straddle_temp(:,2);
    end
    model_hedge_values_lookback(i,:)=hedge_values_lookback_temp(:,1);
	model_hedge_values_straddle(i,:)=hedge_values_straddle_temp(:,1);
end    




plot_hedge_values_day_zero(GOOG_cal_idx(1),GOOG_date,GOOG_S0(1),model_hedge_values_lookback(1,:),model_hedge_values_straddle(1,:), prices, GOOG_day,GOOG_month,GOOG_year)


for i = 2:size(maturities,1)-1
    plot_hedge_values_after_day_zero(GOOG_cal_idx(i),GOOG_date,GOOG_S0(i),model_hedge_values_lookback(i,:),model_hedge_values_straddle(i,:),previous_date_model_hedge_values_lookback(i,:),previous_date_model_hedge_values_straddle(i,:), prices, GOOG_day,GOOG_month,GOOG_year)
end

function [] = plot_hedge_values_after_day_zero(day,GOOG_date,GOOG_S0,hedge_values_lookback,hedge_values_straddle,hedge_values_lookback_prev,hedge_values_straddle_prev, prices, GOOG_day,GOOG_month,GOOG_year)
 
[Iday,~]=find(GOOG_date==day);

   hold off
   xlabel('Price')
   ylabel('Hedge Value')
   plot(prices,hedge_values_lookback,'b-','LineWidth',3)
   hold on
   plot(prices,hedge_values_straddle,'r-','LineWidth',3)
   hold on
   plot(prices,hedge_values_lookback_prev,'b+')
   hold on
   plot(prices,hedge_values_straddle_prev,'r+')
   hold on
   xline(GOOG_S0,'--b');

   xlabel('Asset Price')
   ylabel('Hedge value')
   legend('Lookback Hedge','Straddle Hedge', 'Previous Date Model Lookback Hedge','Previous Date Model Straddle Hedge','ATM','Location','Best')
   plot_title = strcat(' Date:  ', string(GOOG_day(Iday(1))),'.', string(GOOG_month(Iday(1))),'.' ,string(GOOG_year(Iday(1))));
   save_string = strcat('Model_Hedge_Date',string(GOOG_day(Iday(1))),string(GOOG_month(Iday(1))),string(GOOG_year(Iday(1))));
   title(plot_title);
   saveas(gcf,save_string,'png');
   delete(gcf);
   hold off

end

function [] = plot_hedge_values_day_zero(day,GOOG_date,GOOG_S0,hedge_values_lookback,hedge_values_straddle, prices, GOOG_day,GOOG_month,GOOG_year)

[Iday,~]=find(GOOG_date==day);

   hold off
   xlabel('Price')
   ylabel('Hedge Value')
   plot(prices,hedge_values_lookback,'b-','LineWidth',3)
   hold on
   plot(prices,hedge_values_straddle,'r-','LineWidth',3)
   hold on

   
   xline(GOOG_S0,'--b');

   xlabel('Asset Price')
   ylabel('Hedge value')
   legend('Lookback Hedge','Straddle Hedge','ATM','Location','Best')
   plot_title = strcat(' Date:  ', string(GOOG_day(Iday(1))),'.', string(GOOG_month(Iday(1))),'.' ,string(GOOG_year(Iday(1))));
   save_string = strcat('Model_Hedge_Date',string(GOOG_day(Iday(1))),string(GOOG_month(Iday(1))),string(GOOG_year(Iday(1))));
   title(plot_title);
   saveas(gcf,save_string,'png');
   delete(gcf);
   hold off

end

