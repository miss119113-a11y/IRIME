package com.zhou.craft.dto;
import java.time.LocalDate;
public record RequestForm(String userName,String title,String category,double budgetMin,double budgetMax,String description,String preferredMaterial,LocalDate dueDate) {}
