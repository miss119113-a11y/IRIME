package com.zhou.craft.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "craft_order")
public class CraftOrder {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private Long requestId;
    private Long quoteId;
    private String workTitle;
    private String customerName;
    private String creatorName;
    private double amount;
    private String status;
    private int progress;
    private String progressText;
    private LocalDateTime createdAt;
    private LocalDateTime expectedAt;

    public CraftOrder(){}
    public CraftOrder(Long requestId,Long quoteId,String workTitle,String customerName,String creatorName,double amount,int leadDays){
        this.requestId=requestId;this.quoteId=quoteId;this.workTitle=workTitle;this.customerName=customerName;this.creatorName=creatorName;
        this.amount=amount;this.status="制作中";this.progress=20;this.progressText="已确认方案，准备材料";
        this.createdAt=LocalDateTime.now();this.expectedAt=createdAt.plusDays(leadDays);
    }
    public Long getId(){return id;}
    public Long getRequestId(){return requestId;}
    public Long getQuoteId(){return quoteId;}
    public String getWorkTitle(){return workTitle;}
    public String getCustomerName(){return customerName;}
    public String getCreatorName(){return creatorName;}
    public double getAmount(){return amount;}
    public String getStatus(){return status;}
    public int getProgress(){return progress;}
    public String getProgressText(){return progressText;}
    public LocalDateTime getCreatedAt(){return createdAt;}
    public LocalDateTime getExpectedAt(){return expectedAt;}
    public void updateProgress(int progress,String text){
        this.progress=Math.max(0,Math.min(progress,100));this.progressText=text;
        this.status=this.progress>=100?"待确认":"制作中";
    }
    public void complete(){this.progress=100;this.progressText="作品已完成，用户已确认收货";this.status="已完成";}
}
