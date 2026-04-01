import cadquery as cq

result = (cq.Workplane().rect(10, 10)
.workplane(offset = 10).rect(5, 5)
.loft())
cq.exporters.export(result, 'GT.stl')