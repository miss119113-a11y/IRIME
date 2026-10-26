package com.zhou.craft.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "craft_quote")
public class Quote {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private Long requestId;
    private String creatorName;
    private double price;
    private int leadDays;
    @Column(length = 1000)
    private String note;
    private LocalDateTime createdAt;

    public Quote(){}
    public Quote(Long requestId,String creatorName,double price,int leadDays,String note){
        this.requestId=requestId;this.creatorName=creatorName;this.price=price;this.leadDays=leadDays;this.note=note;this.createdAt=LocalDateTime.now();
    }
    public Long getId(){return id;}
    public Long getRequestId(){return requestId;}
    public String getCreatorName(){return creatorName;}
    public double getPrice(){return price;}
    public int getLeadDays(){return leadDays;}
    public String getNote(){return note;}
    public LocalDateTime getCreatedAt(){return createdAt;}
}
