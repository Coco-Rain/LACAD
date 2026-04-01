import cadquery as cq

result = (
cq.Workplane("XY")
.rect(3,3)
.extrude(10,taper=60)
.faces('<Z')
.workplane()
.rect(2.5,2.5)
.cutBlind(-1,taper=60)
)
cq.exporters.export(result, 'GT.stl')