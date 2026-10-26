package com.zhou.craft.model;

import jakarta.persistence.*;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Entity
@Table(name = "custom_request")
public class CustomRequest {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private String userName;
    private String title;
    private String category;
    private double budgetMin;
    private double budgetMax;
    @Column(length = 1600)
    private String description;
    private String preferredMaterial;
    private LocalDate dueDate;
    private String status;
    private LocalDateTime createdAt;

    public CustomRequest(){}
    public CustomRequest(String userName,String title,String category,double budgetMin,double budgetMax,String description,String preferredMaterial,LocalDate dueDate){
        this.userName=userName;this.title=title;this.category=category;this.budgetMin=budgetMin;this.budgetMax=budgetMax;
        this.description=description;this.preferredMaterial=preferredMaterial;this.dueDate=dueDate;
        this.status="待报价";this.createdAt=LocalDateTime.now();
    }
    public Long getId(){return id;}
    public String getUserName(){return userName;}
    public String getTitle(){return title;}
    public String getCategory(){return category;}
    public double getBudgetMin(){return budgetMin;}
    public double getBudgetMax(){return budgetMax;}
    public String getDescription(){return description;}
    public String getPreferredMaterial(){return preferredMaterial;}
    public LocalDate getDueDate(){return dueDate;}
    public String getStatus(){return status;}
    public LocalDateTime getCreatedAt(){return createdAt;}
    public void setStatus(String status){this.status=status;}
}
